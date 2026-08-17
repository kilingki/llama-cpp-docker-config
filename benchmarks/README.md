# Benchmark notes

This directory is a placeholder for later measurements. Do not expect a full harness in the initial runtime.

## Planned comparison axes

- Quantization
- Context length: `32K`, `64K`, `128K`, `192K`, `262K`
- KV cache type (`CACHE_TYPE_K` / `CACHE_TYPE_V`)
- Flash Attention
- batch size
- ubatch size

## Metrics to record

- VRAM usage
- Prompt processing tok/s
- Generation tok/s
- TTFT
- maximum stable context

## How to run a single llama-bench pass

The container must already be up (`docker compose up -d`). `llama-bench` is a separate binary from `llama-server`; unload the model first if you need the full GPU VRAM.

```bash
./scripts/benchmark.sh
```

Optional overrides:

```bash
FLASH_ATTN=on CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 N_PROMPT=2048 ./scripts/benchmark.sh
```

Write outputs under `benchmarks/results/` (gitignored).
