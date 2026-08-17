#!/usr/bin/env bash
# Text load / chat / unload regression.
# Default profile qwen3.8-27b sets MMPROJ_PATH; load requires that file under /models.
# This script does not require an image fixture or GPU residual checks.
# enable_thinking=false so content is not empty while the model is thinking.
# Writes output/<timestamp>-smoke-test/INDEX.md (VRAM polling is optional).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PUBLIC_PORT:-8000}}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"
RECORD="${ROOT}/scripts/record_case.sh"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/output/${STAMP}-smoke-test"
RUN_COMMAND="./scripts/smoke_test.sh"
START_SEC="$(date +%s)"

mkdir -p "${RUN_DIR}"

finish() {
  local code=$?
  trap - EXIT
  local verdict="PASS"
  if [[ "${code}" -ne 0 ]]; then
    verdict="FAIL"
  fi
  local elapsed=$(( $(date +%s) - START_SEC ))
  "${RECORD}" --write-index "${RUN_DIR}" \
    --name smoke-test \
    --command "${RUN_COMMAND}" \
    --verdict "${verdict}" \
    --elapsed "${elapsed}" \
    --model "${MODEL_NAME}" \
    --base-url "${BASE_URL}" >/dev/null
  echo
  echo "results: ${RUN_DIR}/INDEX.md"
  if [[ "${code}" -eq 0 ]]; then
    echo "smoke test passed"
  else
    echo "smoke test failed (see ${RUN_DIR}/INDEX.md)" >&2
  fi
  exit "${code}"
}
trap finish EXIT

step() {
  local dir="$1"
  shift
  echo "== ${dir}"
  "${RECORD}" --out-dir "${RUN_DIR}/${dir}" --base-url "${BASE_URL}" --model "${MODEL_NAME}" "$@"
}

step "01-health-unloaded" --case health-unloaded --mode health --expect-status 200
step "02-models-expect-503" --case models-expect-503 --mode models --expect-status 503
step "03-load" --case load --mode load --expect-status 200
step "04-chat" --case chat --mode chat \
  --prompt "Reply with the single word: pong" \
  --thinking off --max-tokens 16 \
  --expect-status 200 --expect-contains pong
step "05-chat-stream" --case chat-stream --mode chat --stream \
  --prompt "Reply with the single word: pong" \
  --thinking off --max-tokens 16 \
  --expect-status 200 --expect-contains pong
step "06-unload" --case unload --mode unload --expect-status 200
step "07-reload" --case reload --mode load --expect-status 200
