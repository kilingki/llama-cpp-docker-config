# llama.cpp CUDA Docker Runtime

Single-container Docker runtime for GGUF LLM and VLM inference with [llama.cpp](https://github.com/ggml-org/llama.cpp) CUDA. The image contains the runtime only. Model weights stay on the host.

One published base URL is one logical model. The container starts unloaded. `POST /control/load` starts `llama-server`, and OpenAI-compatible `/v1/*` requests are proxied to it. InferSwap uses the same control API as the other runtimes in this set: empty load and unload bodies, and `GET /control/status` as the readiness source.

This repository is [MIT](LICENSE). llama.cpp is [MIT](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE). GGUF weights are licensed separately and are not part of this repository.

## Contents

- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Model control](#model-control)
- [API examples](#api-examples)
- [Configuration](#configuration)
- [Multimodal](#multimodal)
- [Limitations](#limitations)
- [Tests](#tests)
- [Benchmarks](#benchmarks)
- [License](#license)

## Architecture

```text
External Client / InferSwap
    |
    |  OpenAI-compatible API + control API
    |  http://localhost:<PUBLIC_PORT>
    v
controller (FastAPI / uvicorn on :8000)
    |-- GET  /health
    |-- POST /control/load
    |-- POST /control/unload
    |-- GET  /control/status
    |-- /v1/*  --->  proxy
                      |
                      v
                 llama-server 127.0.0.1:8080
                 (child process group, not published)
```

Compose sets `init: true`, so Docker init is PID 1 and uvicorn is the service process. The controller binds `0.0.0.0:8000` inside the container (`Dockerfile` `CMD`). `PUBLIC_PORT` only changes the host publish port.

```text
container lifecycle  !=  model lifecycle

model load    = llama-server process start
model unload  = llama-server process termination
```

The container stays up while unloaded. Model state moves `unloaded` → `loading` → `ready` → `unloading` → `unloaded`. Load or unload failure sets `failed`, together with `residency` `resident`, `not_resident`, or `unknown`. An unexpected llama-server exit sets `failed`. `/health` only reports that the controller process is up. Model readiness is `GET /control/status`.

## Project structure

- `docker-compose.yml`: one `llama-runtime` service
- `prepare-inferswap`: start the container when none exists, or leave an existing container unchanged
- `.env.example`: host port, model directory, and profile name
- `configs/common.env`: llama-server bind, fit mode, and load/unload timeouts
- `configs/models/<name>.env`: one GGUF profile; the default example is `qwen3.8-27b`
- `controller/`: FastAPI control API and `/v1` proxy
- `tests/test_control.py`: control state machine, no GPU
- `tests/smoke_api.py`: health, load, chat, stream, unload against a running container
- `tests/infer_qwen.py`: reasoning-effort and image checks
- `tests/check_gpu_lifecycle.py`: repeated load, chat, and unload with host GPU memory
- `tests/benchmark.sh`: one `llama-bench` pass

## Requirements

- NVIDIA GPU
- NVIDIA driver
- Docker and Docker Compose
- NVIDIA Container Toolkit

The image is built from CUDA 12.8.1 (Ubuntu 24.04) and llama.cpp **b10453**. The default `CUDA_ARCH` is `86` (sm_86, for example an RTX 3090). Other GPUs need a rebuild with a matching `CUDA_ARCH`.

Verified on Windows 11 + WSL2 Ubuntu 24.04 with an RTX 3090 24GB.

## Quick start

1. Copy the example environment file and set the host directory that contains the GGUF files.

```bash
cp .env.example .env
```

```env
PUBLIC_PORT=8000
HOST_MODELS_DIR=/path/to/gguf
MODEL_PROFILE=qwen3.8-27b
```

`HOST_MODELS_DIR` is a host path. Compose mounts it read-only at `/models`. Profile `MODEL_PATH` and optional `MMPROJ_PATH` are container paths under `/models`. The default profile expects both the Qwen3.8-27B GGUF and its mmproj in that directory.

2. Build and start the container. A new container does not load the model.

```bash
docker compose build
docker compose up -d
```

`./prepare-inferswap` does the same start when no container exists. If a container already exists and `GET /control/status` succeeds, the script exits 0 and does not restart or unload it. If the container exists but status fails, it exits non-zero and does not recreate it.

3. Confirm the unloaded state, load the profile, call chat, then unload.

```bash
curl -s http://localhost:8000/control/status
curl -s -X POST http://localhost:8000/control/load \
  -H "Content-Type: application/json" \
  -d '{}'
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
curl -s -X POST http://localhost:8000/control/unload
```

Add another GGUF by creating `configs/models/<name>.env` and setting `MODEL_PROFILE=<name>` in `.env`. A rebuild is not required when only the GGUF file changes. `POST /control/load` does not take a model name.

## Model control

InferSwap talks to this runtime through:

```text
GET  /control/status
POST /control/load
POST /control/unload
/v1/*
```

`GET /health` is the container healthcheck. InferSwap does not use it. The healthcheck passes while the model is unloaded.

Control requests take an empty body or `{}`. Any other JSON object is HTTP 400 `BAD_REQUEST`.

`GET /control/status` returns:

```json
{
  "state": "unloaded",
  "residency": "not_resident",
  "active_requests": 0,
  "last_error": null
}
```

`state` is `unloaded`, `loading`, `ready`, `unloading`, or `failed`. `residency` is `resident`, `not_resident`, or `unknown`. `active_requests` counts accepted inference requests. `GET /v1/models` is not counted. Other `/v1/*` requests count as one until llama-server finishes them, including when the client disconnects first. `last_error` is `null` after a successful load or unload, or `{"code","message"}` after a lifecycle failure.

`POST /control/load` returns 200 with `state=ready` and `residency=resident` when the server can accept inference. Calling it again while ready does not start a second llama-server. `POST /control/unload` returns 200 with `state=unloaded`, `residency=not_resident`, and `active_requests=0`. Calling it again in that state is a no-op. Unload returns 409 `BUSY` while `active_requests` is greater than 0. A load during unload, or an unload during load, returns 409 `LIFECYCLE_CONFLICT`.

Control errors use this body:

```json
{
  "error": {
    "code": "BUSY",
    "message": "runtime has active inference requests"
  }
}
```

`code` is `BAD_REQUEST`, `BUSY`, `LIFECYCLE_CONFLICT`, `LOAD_FAILED`, `UNLOAD_FAILED`, or `STATUS_FAILED`.

If `/v1/*` is called while `state` is not `ready`, the controller returns 503 `{"detail":"model is not ready"}` and does not call llama-server.

A missing model file is HTTP 500 `LOAD_FAILED`. `state` becomes `failed`, `residency` is `not_resident`, and llama-server is not started. Load can be retried.

## API examples

Health while unloaded:

```bash
curl -s http://localhost:8000/health
```

```json
{
  "controller": "ok",
  "llama_server": "stopped"
}
```

Chat. Default Qwen3.8 thinking can leave `message.content` empty. The examples below turn it off.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Streaming:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Image. The controller proxies `/v1/*` without parsing the body. llama.cpp b10453 accepts OpenAI-style `image_url` on `/v1/chat/completions`: `data:image/...;base64,...`, raw base64, or a remote `http(s)://` URL. llama-server fetches remote URLs. This runtime does not download or resize images. Multiple images are extra content parts. Local `file://` is not exposed.

```bash
python3 - <<'PY'
import base64, json, os, urllib.request

path = os.environ.get("IMAGE_FILE", "photo.jpg")
with open(path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
payload = {
    "model": "qwen3.8-27b",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
PY
```

## Configuration

Model values are not hardcoded in Python or the Dockerfile.

```text
.env                         # HOST_MODELS_DIR, MODEL_PROFILE, PUBLIC_PORT
configs/common.env
configs/models/<name>.env    # default example: qwen3.8-27b.env
```

The default example profile is `qwen3.8-27b` (Qwen3.8-27B GGUF Q4_K_M plus a matching mmproj). Default `CONTEXT_SIZE` is `32768`.

Profile knobs map to llama-server flags. The controller also always passes `--no-ui`.

| env | llama-server |
| --- | --- |
| `MODEL_PATH` | `--model` |
| `MODEL_NAME` | `--alias` |
| `CONTEXT_SIZE` | `--ctx-size` |
| `GPU_LAYERS` | `--n-gpu-layers` |
| `CACHE_TYPE_K` | `--cache-type-k` |
| `CACHE_TYPE_V` | `--cache-type-v` |
| `FLASH_ATTN` | `--flash-attn` |
| `BATCH_SIZE` | `--batch-size` |
| `UBATCH_SIZE` | `--ubatch-size` |
| `THREADS` | `--threads` |
| `PARALLEL` | `--parallel` |
| `HOST` | `--host` |
| `PORT` | `--port` |
| `MMPROJ_PATH` | `--mmproj` (optional) |
| `JINJA` | `--jinja` / `--no-jinja` |
| `LLAMA_FIT` | `--fit` (default `off`) |

`LOAD_TIMEOUT_SEC` waits for llama-server `/health` after spawn (default `300`). `UNLOAD_TIMEOUT_SEC` waits after SIGTERM before SIGKILL (default `30`). Those two are controller timeouts, not llama-server flags.

Example host layout under `${HOST_MODELS_DIR}` (filenames are examples; change the env file if yours differ):

```text
/models/
├── Qwen3.8-27B-Q4_K_M.gguf
└── Qwen3.8-27B-mmproj-F16.gguf
```

`mmproj` must match the model architecture and checkpoint. `POST /control/load` does not accept an mmproj field. On or off is the profile only. An empty `MMPROJ_PATH` starts a text-only server.

## Multimodal

Qwen3.8-27B is one model identity. `mmproj` is the llama.cpp projector that turns vision on for that same server process. There is no separate vision profile.

```text
MMPROJ_PATH=
→ text-only llama-server (no --mmproj)

MMPROJ_PATH=/models/xxx-mmproj.gguf
→ same llama-server with multimodal enabled
```

Text chat still works when mmproj is loaded. The default profile sets `MMPROJ_PATH` because this model is natively multimodal. Leave it empty on other profiles for text-only.

Path checks run at `/control/load`, not at controller startup, so an unloaded container can still start.

- missing `MODEL_PATH` file → `LOAD_FAILED`, `MODEL_PATH does not exist: ...`
- non-empty missing `MMPROJ_PATH` → `LOAD_FAILED`, `MMPROJ_PATH does not exist: ...`
- empty `MMPROJ_PATH` → text-only, no mmproj check

## Limitations

- Audio input is out of scope.
- Native video is unsupported. llama.cpp b10453 can accept `input_video`, but decoding needs `ffmpeg`, and this image does not install it. This runtime does not sample frames or transcode video. Send sampled frames as multiple `image_url` parts. CLI `--video` belongs to `llama-mtmd-cli`, not this HTTP runtime.
- Remote image URLs are fetched by llama-server, not the controller. The container needs outbound network access.
- Image tokens use extra context and VRAM. `LLAMA_FIT=off`, so llama-server does not shrink ctx or batch. On OOM, lower `CONTEXT_SIZE` or `BATCH_SIZE` in the profile, or use a smaller mmproj quant.
- A recorded benchmark sweep (context length, quant, KV cache, TTFT) is not implemented. `./tests/benchmark.sh` is one `llama-bench` pass to stdout.
- FastAPI also serves `/docs`, `/redoc`, and `/openapi.json`. The API has no authentication. Keep the published port on a private interface.
- vLLM, TensorRT-LLM, and Triton are out of scope.

## Tests

Contract tests need `controller/requirements.txt` installed and do not need a GPU:

```bash
python3 -m unittest tests.test_control
```

The next commands need a running container.

```bash
python3 tests/smoke_api.py
python3 tests/infer_qwen.py
python3 tests/check_gpu_lifecycle.py
```

- `tests/smoke_api.py`: health while unloaded, `/v1/models` expects 503, load, chat, stream, unload, reload. Chat steps send `enable_thinking: false` and require `pong` in `content`. The last step is reload, so a passing run leaves the model loaded.
- `tests/infer_qwen.py`: reasoning-effort text cases, then image description. It requires `local/test-cat-512.jpg` and `local/test-cat.jpg`, and exits if either file is missing. A passing run leaves the model loaded.
- `tests/smoke_api.py` and `tests/infer_qwen.py` write `tests/outputs/<timestamp>-*/INDEX.md`.
- `tests/check_gpu_lifecycle.py` repeats load, chat, and unload and records host GPU memory in `tests/outputs/gpu_lifecycle.md`. It does not invent an unmeasured byte budget.

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md).

```bash
./tests/benchmark.sh
```

Unload the model first if you need the full GPU VRAM. The wrapper prints one `llama-bench` pass to stdout.

## License

[MIT](LICENSE)
