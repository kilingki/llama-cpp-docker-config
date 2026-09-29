# Benchmark notes

`./tests/benchmark.sh` runs a single `llama-bench` pass inside the already-running container. It is not a full harness. A recorded sweep of quant, context length, KV cache, Flash Attention, and batch size is not implemented.

## How to run a single llama-bench pass

The container must already be up (`docker compose up -d`). `llama-bench` is a separate binary from `llama-server`. Unload the model first if you need the full GPU VRAM.

```bash
./tests/benchmark.sh
```

Optional overrides:

```bash
FLASH_ATTN=on CACHE_TYPE_K=q8_0 CACHE_TYPE_V=q8_0 N_PROMPT=2048 ./tests/benchmark.sh
```

Output goes to stdout. The wrapper does not write result files.
