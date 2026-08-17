# llama.cpp CUDA Docker Runtime

Generic Docker runtime for GGUF LLM/VLM inference with llama.cpp CUDA. The image contains the runtime only. Model weights stay on the host.

Set `HOST_MODELS_DIR` to the host directory that contains GGUF files. Compose mounts that directory read-only at `/models` in the container. Profile `MODEL_PATH` and optional `MMPROJ_PATH` are container paths under `/models`.

## Project purpose

- llama.cpp CUDA backend for GGUF models
- OpenAI-compatible inference API
- InferSwap-oriented load/unload lifecycle API
- Runtime image and GGUF files are separate

vLLM, TensorRT-LLM, and Triton are out of scope. This repository owns the llama.cpp runtime only.

## Requirements

- NVIDIA GPU
- NVIDIA Driver
- Docker
- NVIDIA Container Toolkit

The image is built from CUDA 12.8.1 (Ubuntu 24.04) and llama.cpp **b10453**. Default `CUDA_ARCH=86` (sm_86, e.g. RTX 3090). Other GPUs need a rebuild with a matching `CUDA_ARCH`.

Verified on Windows 11 + WSL2 Ubuntu 24.04 with RTX 3090 24GB.

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

Compose sets `init: true`, so Docker init is PID 1 and uvicorn is the service process. The controller always binds `0.0.0.0:8000` inside the container (`Dockerfile` `CMD`). `PUBLIC_PORT` only changes the host publish port.

```text
container lifecycle  !=  model lifecycle

model load    = llama-server process start
model unload  = llama-server process termination
```

The container stays up while unloaded. InferSwap does not need llama.cpp internals; it uses the controller contract only.

Model state: `UNLOADED` → `LOADING` → `READY` → `UNLOADING` → `UNLOADED`. Unexpected llama-server exit sets `ERROR`. `/health` `llama_server` is `stopped` | `loading` | `ok` | `stopping` | `error`.

## InferSwap integration

InferSwap talks to this runtime through:

```text
POST /control/load
POST /control/unload
GET  /control/status
GET  /health
/v1/*
```

## Usage

```text
cp .env.example .env
# set HOST_MODELS_DIR to the host directory that contains GGUF files
# default profile qwen3.8-27b expects both the GGUF and its mmproj under that directory
docker compose build
docker compose up -d
curl -X POST http://localhost:8000/control/load
curl http://localhost:8000/v1/chat/completions ...
curl -X POST http://localhost:8000/control/unload
```

Add another GGUF by creating `configs/models/<name>.env` and setting `MODEL_PROFILE=<name>` in `.env`. Rebuild is not required when only the GGUF file changes. Switching models at runtime via the load body is not implemented; see [Roadmap](#roadmap).

## API example

Examples below use the default profile `qwen3.8-27b`.

Health while unloaded:

```bash
curl -s http://localhost:8000/health
```

```json
{
  "controller": "ok",
  "model_state": "UNLOADED",
  "llama_server": "stopped"
}
```

Load the configured profile. `model` is optional; if set, it must match `MODEL_NAME` or the controller returns `400`.

```bash
curl -s -X POST http://localhost:8000/control/load \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b"}'
```

```json
{
  "status": "loaded",
  "state": "READY",
  "model": "qwen3.8-27b",
  "pid": 123
}
```

A second load while already `READY` returns `200` with `"status": "already_loaded"`. Load or unload while `LOADING` / `UNLOADING` returns `409`.

Status:

```bash
curl -s http://localhost:8000/control/status
```

```json
{
  "state": "READY",
  "model": "qwen3.8-27b",
  "pid": 123,
  "backend": "llama.cpp",
  "runtime_port": 8080
}
```

Chat completion. Default Qwen3.8 thinking can leave `message.content` empty; smoke and the examples below turn it off.

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

Unload:

```bash
curl -s -X POST http://localhost:8000/control/unload
```

```json
{
  "status": "unloaded",
  "state": "UNLOADED"
}
```

Already unloaded returns `200` with `"status": "already_unloaded"`.

If `/v1/*` is called while the model is not `READY`, the controller returns `503` with `{"detail":"Model is not loaded"}` instead of leaking an internal connection error.

## Multimodal

Qwen3.8-27B is one model identity. `mmproj` is the llama.cpp projector that turns vision on for that same server process. There is no `qwen3.8-27b-vl` profile.

```text
Qwen3.8-27B
├── Text
├── Image (with mmproj)
└── Video (unsupported in this image; send frames as image_url)
```

```text
MMPROJ_PATH=
→ text-only llama-server (no --mmproj)

MMPROJ_PATH=/models/xxx-mmproj.gguf
→ same llama-server with multimodal enabled
```

The default profile sets `MMPROJ_PATH` because this model is natively multimodal. Leave it empty on other profiles for text-only. `POST /control/load` does not accept an mmproj field; on/off is env/profile only.

Example host layout under `${HOST_MODELS_DIR}` (filenames are examples; change env if yours differ):

```text
/models/
├── Qwen3.8-27B-Q4_K_M.gguf
└── Qwen3.8-27B-mmproj-F16.gguf
```

`mmproj` must match the model architecture and checkpoint. It is not a generic file shared across unrelated GGUFs.

Path checks run at `/control/load`, not controller startup, so an unloaded container can still start. Failures are HTTP `500` and llama-server is not started:

- missing `MODEL_PATH` file → `{"detail":"Model load failed: MODEL_PATH does not exist: ..."}`
- non-empty missing `MMPROJ_PATH` → `{"detail":"Model load failed: MMPROJ_PATH does not exist: ..."}`
- empty `MMPROJ_PATH` → text-only, no mmproj check

To confirm a missing mmproj fails load, start the container with `MMPROJ_PATH=/models/does-not-exist.gguf` and `POST /control/load`. Do not put the path on the load request body.

### Text inference

Same as the chat example above. Text still works when mmproj is loaded.

### Image inference

The controller proxies `/v1/*` without parsing the body. llama.cpp b10453 accepts OpenAI-style `image_url` on `/v1/chat/completions`: `data:image/...;base64,...`, raw base64, or a remote `http(s)://` URL (llama-server fetches it; this runtime does not download or resize images). Multiple images are extra content parts. Local `file://` is not exposed (`--media-path` is not configured).

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

### Video

llama.cpp b10453 can accept `input_video` on `/v1/chat/completions` (`data` or `url`), but decoding needs `ffmpeg` on the llama-server PATH. This image does not install ffmpeg, and this runtime does not sample frames or transcode video. Treat native video as unsupported here; see [Roadmap](#roadmap).

Send sampled frames as multiple `image_url` parts from the caller instead. CLI `--video` belongs to `llama-mtmd-cli`, not this HTTP runtime.

### Limitations

- Audio input is out of scope.
- Remote image URLs are fetched by llama-server, not the controller. The container needs outbound network access.
- Image tokens use extra context and VRAM. This runtime uses `LLAMA_FIT=off`, so llama-server will not silently shrink ctx/batch. If you OOM, lower `CONTEXT_SIZE` / `BATCH_SIZE` in the profile or use a smaller mmproj quant.
- After unload, check leftover projector CUDA memory with `nvidia-smi` if needed. That check is not part of smoke.

## Configuration

Model values are not hardcoded in Python or the Dockerfile.

```text
.env                         # HOST_MODELS_DIR, MODEL_PROFILE, PUBLIC_PORT
configs/common.env
configs/models/<name>.env    # default example: qwen3.8-27b.env
```

`HOST_MODELS_DIR` is a host path. `MODEL_PATH` / `MMPROJ_PATH` are container paths under `/models`.

The default example profile is `qwen3.8-27b` (Qwen3.8-27B GGUF Q4_K_M plus matching mmproj). Default `CONTEXT_SIZE` is `32768`; do not treat 262K as the first-run setting.

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

Controller timeouts (not llama-server flags): `LOAD_TIMEOUT_SEC` waits for llama-server `/health` after spawn (default `300`); `UNLOAD_TIMEOUT_SEC` waits after SIGTERM before SIGKILL (default `30`).

## Tests

```bash
./scripts/smoke_test.sh
IMAGE_FILE=/path/to/photo.jpg ./scripts/smoke_vision.sh
```

- `smoke_test.sh`: health (unloaded) → `/v1/models` expects `503` → load → chat → stream → unload → reload. Chat steps send `enable_thinking: false` and require `pong` in `content`. The last step is reload, so a passing run leaves the model loaded.
- `smoke_vision.sh`: load → text-with-mmproj. `IMAGE_FILE` is optional; when set, an image chat step runs. Without it, the script still passes after the text step.
- Each run writes `output/<timestamp>-smoke-*/INDEX.md` (request/response per step, optional VRAM peaks when `nvidia-smi` is available). `RECORD_VRAM=0` skips VRAM polling. The default text smoke does not require an image fixture or GPU tools.
- `scripts/record_case.sh` is the internal recorder those scripts call. It always encodes local images as `data:image/jpeg;base64,...`.

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md). `./scripts/benchmark.sh` is a thin `llama-bench` wrapper (one pass, stdout). Unload the model first if you need the full GPU VRAM. A recorded sweep is not implemented; see [Roadmap](#roadmap).

## Roadmap

Not implemented. Checkboxes are intended work, not current behavior.

- [ ] Runtime model switching (`POST /control/load` `model` must match the configured profile today)
- [ ] Recorded benchmark harness (VRAM, prompt tok/s, generation tok/s, TTFT, maximum stable context)
- [ ] Context-length sweep: 32K, 64K, 128K, 192K, 262K
- [ ] MTP / speculative decoding
- [ ] Native video in this image (`input_video` + ffmpeg)
