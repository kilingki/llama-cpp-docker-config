# Benchmark notes

`./scripts/benchmark.sh` runs a single `llama-bench` pass inside the already-running container. It is not a full harness.

Recorded sweeps (quant, context length, KV cache, Flash Attention, batch/ubatch) and saved metrics live on the [root README Roadmap](../README.md#roadmap).

## How to run a single llama-bench pass

The container must already be up (`docker compose up -d`). `llama-bench` is a separate binary from `llama-server`; unload the model first if you need the full GPU VRAM.

```bash
./scripts/benchmark.sh
```

Optional overrides:

```bash
FLASH_ATTN=on CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 N_PROMPT=2048 ./scripts/benchmark.sh
```

Output goes to stdout. `benchmarks/results/` is gitignored for later saved runs; the wrapper does not write files there yet.
