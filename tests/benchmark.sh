#!/usr/bin/env bash
# Thin wrapper around llama-bench for later comparison of:
#   quantization, context length, KV cache type, flash attention, batch, ubatch
# This script does not run a full sweep. Pass llama-bench flags after -- .
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -z "${MODEL_PATH_CONTAINER:-}" ]]; then
  if [[ -n "${MODEL_PATH_HOST:-}" ]]; then
    MODEL_PATH_CONTAINER="/models/$(basename "${MODEL_PATH_HOST}")"
  else
    MODEL_PATH_CONTAINER="/models/Qwen3.8-27B-Q4_K_M.gguf"
  fi
fi
SERVICE="${SERVICE:-llama-runtime}"

N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
N_PROMPT="${N_PROMPT:-512}"
N_GEN="${N_GEN:-128}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-md}"

exec docker compose exec -T "${SERVICE}" llama-bench \
  --model "${MODEL_PATH_CONTAINER}" \
  --n-gpu-layers "${N_GPU_LAYERS}" \
  --flash-attn "${FLASH_ATTN}" \
  --cache-type-k "${CACHE_TYPE_K}" \
  --cache-type-v "${CACHE_TYPE_V}" \
  --batch-size "${BATCH_SIZE}" \
  --ubatch-size "${UBATCH_SIZE}" \
  --n-prompt "${N_PROMPT}" \
  --n-gen "${N_GEN}" \
  --output "${OUTPUT_FORMAT}" \
  "$@"
