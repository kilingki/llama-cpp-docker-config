#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:${PUBLIC_PORT:-8000}}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing dependency: $1" >&2
    exit 1
  fi
}

need curl
need python3

echo "== GET ${BASE_URL}/health (expect UNLOADED)"
curl -fsS "${BASE_URL}/health"
echo

echo "== GET ${BASE_URL}/control/status"
curl -fsS "${BASE_URL}/control/status"
echo

echo "== GET ${BASE_URL}/v1/models (expect 503)"
code="$(curl -sS -o /tmp/llama-smoke-v1.json -w '%{http_code}' "${BASE_URL}/v1/models" || true)"
cat /tmp/llama-smoke-v1.json
echo
if [[ "${code}" != "503" ]]; then
  echo "expected 503 before load, got ${code}" >&2
  exit 1
fi

echo "== POST ${BASE_URL}/control/load"
curl -fsS -X POST "${BASE_URL}/control/load" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\"}"
echo

echo "== GET ${BASE_URL}/health (expect READY)"
curl -fsS "${BASE_URL}/health"
echo

echo "== GET ${BASE_URL}/v1/models"
curl -fsS "${BASE_URL}/v1/models"
echo

echo "== POST ${BASE_URL}/v1/chat/completions"
curl -fsS "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],\"max_tokens\":16,\"temperature\":0}"
echo

echo "== POST ${BASE_URL}/v1/chat/completions (streaming)"
curl -fsS -N "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: pong\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}"
echo

echo "== POST ${BASE_URL}/control/unload"
curl -fsS -X POST "${BASE_URL}/control/unload"
echo

echo "== GET ${BASE_URL}/health (expect UNLOADED)"
curl -fsS "${BASE_URL}/health"
echo

echo "== POST ${BASE_URL}/control/load (reload)"
curl -fsS -X POST "${BASE_URL}/control/load" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\"}"
echo

echo "== GET ${BASE_URL}/control/status"
curl -fsS "${BASE_URL}/control/status"
echo

echo "smoke test passed"
