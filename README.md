# llama.cpp CUDA Docker Runtime

Generic Docker runtime for GGUF LLM/VLM inference with llama.cpp CUDA. The image contains the runtime only. Model weights stay on the host.

Set `HOST_MODELS_DIR` to the host directory that contains GGUF files. Compose mounts that directory read-only at `/models` in the container. Profile `MODEL_PATH` is a container path under `/models`.

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

Verified on Windows 11 + WSL2 Ubuntu 24.04 with RTX 3090 24GB.

## Architecture

```text
External Client / InferSwap
    |
    |  OpenAI-compatible API + control API
    |  http://localhost:<PUBLIC_PORT>
    v
controller (PID 1, FastAPI)
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

```text
container lifecycle  !=  model lifecycle

model load    = llama-server process start
model unload  = llama-server process termination
```

The container stays up while unloaded. InferSwap does not need llama.cpp internals; it uses the controller contract only.

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
# for the default profile, that directory should include Qwen3.8-27B-Q4_K_M.gguf
docker compose build
docker compose up -d
curl -X POST http://localhost:8000/control/load
curl http://localhost:8000/v1/chat/completions ...
curl -X POST http://localhost:8000/control/unload
```

Add another GGUF by creating `configs/models/<name>.env` and setting `MODEL_PROFILE=<name>` in `.env`. Rebuild is not required when only the GGUF file changes.

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

Load the configured profile:

```bash
curl -s -X POST http://localhost:8000/control/load \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b"}'
```

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

Chat completion:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
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
    "stream": true
  }'
```

Unload:

```bash
curl -s -X POST http://localhost:8000/control/unload
```

If `/v1/*` is called while the model is not READY, the controller returns `503` with `{"detail":"Model is not loaded"}` instead of leaking an internal connection error.

## Configuration

Model values are not hardcoded in Python or the Dockerfile.

```text
.env                         # HOST_MODELS_DIR, MODEL_PROFILE, PUBLIC_PORT
configs/common.env
configs/models/<name>.env    # default example: qwen3.8-27b.env
```

`HOST_MODELS_DIR` is a host path. `MODEL_PATH` / `MMPROJ_PATH` are container paths under `/models`.

The default example profile is `qwen3.8-27b` (Qwen3.8-27B GGUF Q4_K_M). Default `CONTEXT_SIZE` is `32768`; do not treat 262K as the first-run setting.

Profile knobs map to llama-server flags:

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

MTP / speculative decoding is not enabled in this version.

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md). Planned recorded metrics:

```text
VRAM usage
Prompt processing tok/s
Generation tok/s
TTFT
maximum stable context
```

Planned context lengths: `32K`, `64K`, `128K`, `192K`, `262K`.

```bash
./scripts/smoke_test.sh
./scripts/benchmark.sh
```
