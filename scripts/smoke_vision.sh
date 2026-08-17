#!/usr/bin/env bash
# Optional mmproj smoke, separate from scripts/smoke_test.sh.
# Assumes the default profile is already loaded (or loads it).
# Image request runs only when IMAGE_FILE is set.
# enable_thinking=false so content is filled; HTTP 200 alone is not enough.
# Writes output/<timestamp>-smoke-vision/INDEX.md (VRAM polling is optional).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PUBLIC_PORT:-8000}}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"
IMAGE_FILE="${IMAGE_FILE:-}"
RECORD="${ROOT}/scripts/record_case.sh"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${ROOT}/output/${STAMP}-smoke-vision"
if [[ -n "${IMAGE_FILE}" ]]; then
  RUN_COMMAND="IMAGE_FILE=${IMAGE_FILE} ./scripts/smoke_vision.sh"
else
  RUN_COMMAND="./scripts/smoke_vision.sh"
fi
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
    --name smoke-vision \
    --command "${RUN_COMMAND}" \
    --verdict "${verdict}" \
    --elapsed "${elapsed}" \
    --model "${MODEL_NAME}" \
    --base-url "${BASE_URL}" >/dev/null
  echo
  echo "results: ${RUN_DIR}/INDEX.md"
  if [[ "${code}" -eq 0 ]]; then
    if [[ -z "${IMAGE_FILE}" ]]; then
      echo "vision smoke (text only) passed"
    else
      echo "vision smoke passed"
      echo "non-empty content confirms the model returned a visible answer; inspect it to judge image understanding."
    fi
  else
    echo "vision smoke failed (see ${RUN_DIR}/INDEX.md)" >&2
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

step "01-load" --case load --mode load --expect-status 200
step "02-text-with-mmproj" --case text-with-mmproj --mode chat \
  --prompt "Reply with the single word: pong" \
  --thinking off --max-tokens 16 \
  --expect-status 200 --expect-contains pong --expect-nonempty

if [[ -z "${IMAGE_FILE}" ]]; then
  echo "IMAGE_FILE unset; skipping image request"
  exit 0
fi

if [[ ! -f "${IMAGE_FILE}" ]]; then
  echo "IMAGE_FILE does not exist: ${IMAGE_FILE}" >&2
  exit 1
fi

step "03-image" --case image --mode chat \
  --prompt "Describe this image in one sentence." \
  --image "${IMAGE_FILE}" \
  --thinking off --max-tokens 128 \
  --expect-status 200 --expect-nonempty
