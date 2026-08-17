#!/usr/bin/env bash
# Record one HTTP step under a directory (smoke helper; also usable ad-hoc).
# VRAM polling is optional: skipped when nvidia-smi is missing or RECORD_VRAM=0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

COMMAND="$(printf '%q ' "$0" "$@")"
COMMAND="${COMMAND% }"

BASE_URL="${BASE_URL:-http://127.0.0.1:${PUBLIC_PORT:-8000}}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b}"
CASE_NAME=""
OUT_DIR=""
MODE="chat"
PROMPT=""
IMAGE_FILE=""
IMAGE_URL=""
MAX_TOKENS="64"
THINKING=""
STREAM="false"
POLL_SEC="0.2"
EXPECT_STATUS=""
EXPECT_CONTAINS=""
EXPECT_NONEMPTY="false"
WRITE_INDEX=""
INDEX_NAME=""
INDEX_COMMAND=""
INDEX_VERDICT=""
INDEX_ELAPSED=""

usage() {
  cat <<'EOF'
Usage: scripts/record_case.sh --case NAME [options]
       scripts/record_case.sh --out-dir DIR [options]
       scripts/record_case.sh --write-index DIR --name NAME [options]

  --case NAME           output/<timestamp>-NAME/ (ad-hoc)
  --out-dir DIR         write into an existing/parent-created directory
  --mode chat|load|unload|snapshot|health|models
  --prompt TEXT         chat only; omitted for load/unload/snapshot/health/models
  --image PATH          local file -> image_url data URL (copied into out-dir)
  --image-url URL       remote http(s) image_url (llama-server fetches)
  --max-tokens N
  --thinking on|off     omit to use model default
  --stream
  --expect-status N     record anyway; exit 1 on mismatch
  --expect-contains TEXT
  --expect-nonempty     chat content must be non-empty
  --base-url URL
  --model NAME
  --write-index DIR     build DIR/INDEX.md and append output/INDEX.md
  --name NAME           run title for --write-index
  --command TEXT        reproduction command for --write-index
  --verdict PASS|FAIL   override computed verdict for --write-index
  --elapsed SEC         wall time for --write-index
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case) CASE_NAME="${2:?}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?}"; shift 2 ;;
    --mode) MODE="${2:?}"; shift 2 ;;
    --prompt) PROMPT="${2:?}"; shift 2 ;;
    --image) IMAGE_FILE="${2:?}"; shift 2 ;;
    --image-url) IMAGE_URL="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --thinking) THINKING="${2:?}"; shift 2 ;;
    --stream) STREAM="true"; shift ;;
    --expect-status) EXPECT_STATUS="${2:?}"; shift 2 ;;
    --expect-contains) EXPECT_CONTAINS="${2:?}"; shift 2 ;;
    --expect-nonempty) EXPECT_NONEMPTY="true"; shift ;;
    --base-url) BASE_URL="${2:?}"; shift 2 ;;
    --model) MODEL_NAME="${2:?}"; shift 2 ;;
    --write-index) WRITE_INDEX="${2:?}"; shift 2 ;;
    --name) INDEX_NAME="${2:?}"; shift 2 ;;
    --command) INDEX_COMMAND="${2:?}"; shift 2 ;;
    --verdict) INDEX_VERDICT="${2:?}"; shift 2 ;;
    --elapsed) INDEX_ELAPSED="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing dependency: $1" >&2
    exit 1
  fi
}

need python3

if [[ -n "${WRITE_INDEX}" ]]; then
  python3 - "${WRITE_INDEX}" "${INDEX_NAME}" "${INDEX_COMMAND}" "${INDEX_VERDICT}" \
    "${INDEX_ELAPSED}" "${MODEL_NAME}" "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
name = sys.argv[2] or run_dir.name
command = sys.argv[3]
verdict_override = sys.argv[4]
elapsed_s = sys.argv[5]
model_name = sys.argv[6]
root = Path(sys.argv[7])

def cell(value) -> str:
    text = "" if value is None else str(value).replace("\n", " ").replace("|", "\\|").strip()
    if len(text) > 80:
        text = text[:77] + "..."
    return text or "—"

steps = []
for child in sorted(p for p in run_dir.iterdir() if p.is_dir()):
    meta_path = child / "meta.json"
    if not meta_path.exists():
        continue
    meta = json.loads(meta_path.read_text())
    steps.append((child.name, meta))

computed = "PASS"
for _, meta in steps:
    if meta.get("verdict") != "PASS":
        computed = "FAIL"
        break
verdict = verdict_override or computed or "PASS"

model = model_name
if steps:
    model = steps[0][1].get("model") or model

elapsed = elapsed_s
if not elapsed:
    total = 0.0
    any_elapsed = False
    for _, meta in steps:
        if meta.get("elapsed_sec") is not None:
            total += float(meta["elapsed_sec"])
            any_elapsed = True
    elapsed = f"{round(total, 3)}" if any_elapsed else "—"

lines = [
    f"# {name}",
    "",
    f"- command: `{command or '—'}`",
    f"- model: `{model}`",
    f"- verdict: `{verdict}`",
    f"- elapsed_sec: `{elapsed}`",
    "",
    "| step | mode | http | input | output | vram peak | elapsed | verdict |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
]
for step_name, meta in steps:
    vram = meta.get("vram_peak_mib")
    lines.append(
        "| "
        + " | ".join(
            [
                cell(step_name),
                cell(meta.get("mode")),
                cell(meta.get("http_status")),
                cell(meta.get("input_summary")),
                cell(meta.get("output_summary")),
                cell(vram if vram is not None else "—"),
                cell(meta.get("elapsed_sec")),
                cell(meta.get("verdict") or "—"),
            ]
        )
        + " |"
    )
if not steps:
    lines.append("| — | — | — | (no steps recorded) | — | — | — | — |")
lines.append("")
(run_dir / "INDEX.md").write_text("\n".join(lines) + "\n")

index_path = root / "output" / "INDEX.md"
index_path.parent.mkdir(parents=True, exist_ok=True)
row = (
    f"| `{run_dir.name}` | `{verdict}` | `{model}` | `{command or '—'}` | `{elapsed}` |\n"
)
if not index_path.exists():
    index_path.write_text(
        "# smoke runs\n\n"
        "| run | verdict | model | command | elapsed |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
with index_path.open("a") as f:
    f.write(row)
print(run_dir / "INDEX.md")
print(verdict)
PY
  exit 0
fi

if [[ -z "${OUT_DIR}" && -z "${CASE_NAME}" ]]; then
  echo "--case or --out-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  OUT_DIR="${ROOT}/output/${STAMP}-${CASE_NAME}"
fi
if [[ -z "${CASE_NAME}" ]]; then
  CASE_NAME="$(basename "${OUT_DIR}")"
fi
mkdir -p "${OUT_DIR}"

IMAGE_COPIED=""
if [[ -n "${IMAGE_FILE}" ]]; then
  if [[ ! -f "${IMAGE_FILE}" ]]; then
    echo "image file does not exist: ${IMAGE_FILE}" >&2
    exit 1
  fi
  IMAGE_COPIED="${OUT_DIR}/$(basename "${IMAGE_FILE}")"
  cp -f "${IMAGE_FILE}" "${IMAGE_COPIED}"
  IMAGE_FILE="${IMAGE_COPIED}"
fi

RECORD_VRAM="${RECORD_VRAM:-1}"
VRAM_ENABLED="false"
POLL_PID=""
SAMPLES="${OUT_DIR}/vram-samples.csv"

cleanup() {
  if [[ -n "${POLL_PID}" ]]; then
    kill "${POLL_PID}" 2>/dev/null || true
    wait "${POLL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${RECORD_VRAM}" != "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  VRAM_ENABLED="true"
  echo "timestamp,memory_used_mib,memory_total_mib,utilization_gpu" > "${SAMPLES}"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits >> "${SAMPLES}" || true
      sleep "${POLL_SEC}"
    done
  ) &
  POLL_PID=$!
  sleep 0.3
fi

python3 - "${OUT_DIR}" "${BASE_URL}" "${MODEL_NAME}" "${MODE}" "${PROMPT}" \
  "${IMAGE_FILE}" "${IMAGE_URL}" "${MAX_TOKENS}" "${THINKING}" "${STREAM}" \
  "${CASE_NAME}" "${COMMAND}" "${EXPECT_STATUS}" "${EXPECT_CONTAINS}" \
  "${EXPECT_NONEMPTY}" "${VRAM_ENABLED}" <<'PY'
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

(
    out_dir,
    base_url,
    model_name,
    mode,
    prompt,
    image_file,
    image_url,
    max_tokens_s,
    thinking,
    stream_s,
    case_name,
    command,
    expect_status_s,
    expect_contains,
    expect_nonempty_s,
    vram_enabled_s,
) = sys.argv[1:]

out = Path(out_dir)
max_tokens = int(max_tokens_s)
stream = stream_s == "true"
expect_nonempty = expect_nonempty_s == "true"
vram_enabled = vram_enabled_s == "true"
expect_status = int(expect_status_s) if expect_status_s else None
t0 = time.time()

image_meta = None
payload = None
http_status = None
response_obj = None
response_text = ""
method = None
url = None


def chat_payload():
    content = prompt
    meta = None
    if image_file:
        raw = Path(image_file).read_bytes()
        import base64

        b64 = base64.b64encode(raw).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            },
        ]
        meta = {
            "image_file": os.path.basename(image_file),
            "image_path": os.path.abspath(image_file),
            "image_bytes": len(raw),
            "image_sha256": hashlib.sha256(raw).hexdigest(),
        }
    elif image_url:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        meta = {"image_url": image_url}
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if thinking == "off":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking == "on":
        body["chat_template_kwargs"] = {"enable_thinking": True}
    return body, meta


def http_json(method, url, data=None, timeout=300):
    raw = None if data is None else json.dumps(data).encode()
    headers = {"Content-Type": "application/json"} if raw else {}
    req = urllib.request.Request(url, data=raw, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def parse_sse(raw_text: str) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    last_obj = None
    for line in raw_text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        last_obj = obj
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning_parts.append(delta["reasoning_content"])
    result = {
        "stream": True,
        "raw_sse": raw_text,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "reasoning_content": "".join(reasoning_parts),
                }
            }
        ],
        "last_chunk": last_obj,
    }
    if isinstance(last_obj, dict):
        if last_obj.get("usage"):
            result["usage"] = last_obj["usage"]
        if last_obj.get("timings"):
            result["timings"] = last_obj["timings"]
    return result


timeouts = {
    "chat": 300,
    "load": 360,
    "unload": 60,
    "snapshot": 10,
    "health": 10,
    "models": 10,
}

if mode == "chat":
    method, url = "POST", f"{base_url}/v1/chat/completions"
    payload, image_meta = chat_payload()
    http_status, body = http_json(method, url, payload, timeout=timeouts[mode])
    response_text = body.decode("utf-8", errors="replace")
    if stream:
        response_obj = parse_sse(response_text)
        (out / "response.sse").write_text(response_text)
    else:
        try:
            response_obj = json.loads(response_text)
        except json.JSONDecodeError:
            response_obj = None
elif mode == "load":
    method, url = "POST", f"{base_url}/control/load"
    payload = {"model": model_name}
    http_status, body = http_json(method, url, payload, timeout=timeouts[mode])
    response_text = body.decode("utf-8", errors="replace")
    try:
        response_obj = json.loads(response_text)
    except json.JSONDecodeError:
        response_obj = None
elif mode == "unload":
    method, url = "POST", f"{base_url}/control/unload"
    payload = {}
    http_status, body = http_json(method, url, payload, timeout=timeouts[mode])
    response_text = body.decode("utf-8", errors="replace")
    try:
        response_obj = json.loads(response_text)
    except json.JSONDecodeError:
        response_obj = None
elif mode == "snapshot":
    method, url = "GET", f"{base_url}/control/status"
    http_status, body = http_json(method, url, timeout=timeouts[mode])
    response_text = body.decode("utf-8", errors="replace")
    try:
        response_obj = json.loads(response_text)
    except json.JSONDecodeError:
        response_obj = None
elif mode == "health":
    method, url = "GET", f"{base_url}/health"
    http_status, body = http_json(method, url, timeout=timeouts[mode])
    response_text = body.decode("utf-8", errors="replace")
    try:
        response_obj = json.loads(response_text)
    except json.JSONDecodeError:
        response_obj = None
elif mode == "models":
    method, url = "GET", f"{base_url}/v1/models"
    http_status, body = http_json(method, url, timeout=timeouts[mode])
    response_text = body.decode("utf-8", errors="replace")
    try:
        response_obj = json.loads(response_text)
    except json.JSONDecodeError:
        response_obj = None
else:
    raise SystemExit(f"unknown mode: {mode}")

elapsed = round(time.time() - t0, 3)
if vram_enabled:
    time.sleep(0.4)

request_record = {
    "mode": mode,
    "method": method,
    "url": url,
    "base_url": base_url,
}
if payload is not None:
    sanitized = json.loads(json.dumps(payload))
    msgs = sanitized.get("messages")
    if isinstance(msgs, list):
        for msg in msgs:
            parts = msg.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if part.get("type") != "image_url":
                    continue
                part_url = (part.get("image_url") or {}).get("url") or ""
                if part_url.startswith("data:"):
                    part["image_url"]["url"] = "data:image/jpeg;base64,<omitted>"
    request_record["payload"] = sanitized
if image_meta:
    request_record.update(image_meta)

content = ""
reasoning = ""
error = None
if isinstance(response_obj, dict):
    choices = response_obj.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        delta = (choices[0] or {}).get("delta") or {}
        if not content:
            content = delta.get("content") or content
        if not reasoning:
            reasoning = delta.get("reasoning_content") or reasoning
    detail = response_obj.get("detail")
    err_obj = response_obj.get("error")
    if isinstance(detail, str) and detail.strip():
        error = detail
    elif isinstance(err_obj, dict) and err_obj.get("message"):
        error = str(err_obj["message"])
    elif isinstance(err_obj, str) and err_obj.strip():
        error = err_obj
    elif http_status and http_status >= 400 and not content:
        error = response_text.strip()[:500] or f"HTTP {http_status}"

(out / "request.json").write_text(json.dumps(request_record, indent=2, ensure_ascii=False) + "\n")
if response_obj is not None:
    to_store = json.loads(json.dumps(response_obj))
    if isinstance(to_store, dict) and "raw_sse" in to_store:
        to_store["raw_sse"] = "<see response.sse>"
    (out / "response.json").write_text(json.dumps(to_store, indent=2, ensure_ascii=False) + "\n")
else:
    (out / "response.json").write_text(response_text + "\n")

vram = None
samples_path = out / "vram-samples.csv"
if vram_enabled and samples_path.exists():
    rows = []
    with samples_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            used = (row.get("memory_used_mib") or "").strip()
            total = (row.get("memory_total_mib") or "").strip()
            try:
                used_i = int(float(used))
                total_i = int(float(total))
            except ValueError:
                continue
            rows.append((used_i, total_i))
    used_vals = [r[0] for r in rows]
    vram = {
        "sample_count": len(rows),
        "before_mib": used_vals[0] if used_vals else None,
        "after_mib": used_vals[-1] if used_vals else None,
        "peak_mib": max(used_vals) if used_vals else None,
        "total_mib": rows[0][1] if rows else None,
    }
    (out / "vram.json").write_text(json.dumps(vram, indent=2) + "\n")

usage = None
timings = None
if isinstance(response_obj, dict):
    usage = response_obj.get("usage")
    timings = response_obj.get("timings")
    last_chunk = response_obj.get("last_chunk")
    if isinstance(last_chunk, dict):
        usage = usage or last_chunk.get("usage")
        timings = timings or last_chunk.get("timings")


def one_line(value, limit=80) -> str:
    text = " ".join(str(value).split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


if mode == "chat":
    input_summary = prompt or "(empty prompt)"
    if image_meta and image_meta.get("image_file"):
        input_summary = f"{input_summary} + image {image_meta['image_file']}"
    elif image_meta and image_meta.get("image_url"):
        input_summary = f"{input_summary} + {image_meta['image_url']}"
    output_summary = error or (content.strip() if content.strip() else "(empty content)")
elif mode == "health":
    input_summary = "GET /health"
    if isinstance(response_obj, dict):
        output_summary = response_obj.get("model_state") or one_line(response_obj)
    else:
        output_summary = error or one_line(response_text)
elif mode == "models":
    input_summary = "GET /v1/models"
    output_summary = error or one_line(response_text if http_status != 200 else response_obj)
elif mode == "load":
    input_summary = f"POST /control/load model={model_name}"
    output_summary = error or one_line(response_obj if response_obj is not None else response_text)
elif mode == "unload":
    input_summary = "POST /control/unload"
    output_summary = error or one_line(response_obj if response_obj is not None else response_text)
elif mode == "snapshot":
    input_summary = "GET /control/status"
    if isinstance(response_obj, dict):
        output_summary = response_obj.get("state") or one_line(response_obj)
    else:
        output_summary = error or one_line(response_text)
else:
    input_summary = mode
    output_summary = error or one_line(response_text)

fail_reasons = []
if expect_status is not None and http_status != expect_status:
    fail_reasons.append(f"expected HTTP {expect_status}, got {http_status}")
if expect_nonempty and not (content or "").strip():
    fail_reasons.append("expected non-empty message.content")
if expect_contains:
    needle = expect_contains.lower()
    haystack = content if mode == "chat" else response_text
    if needle not in (haystack or "").lower():
        fail_reasons.append(f"expected {expect_contains!r} in {'content' if mode == 'chat' else 'response'}")
verdict = "FAIL" if fail_reasons else "PASS"

meta = {
    "case_dir": str(out),
    "case": case_name,
    "model": model_name,
    "mode": mode,
    "command": command,
    "http_status": http_status,
    "elapsed_sec": elapsed,
    "verdict": verdict,
    "fail_reasons": fail_reasons,
    "input_summary": one_line(input_summary, 120),
    "output_summary": one_line(output_summary, 120),
    "vram_peak_mib": (vram or {}).get("peak_mib") if vram else None,
}
if mode == "chat":
    meta.update(
        {
            "prompt": prompt,
            "thinking": thinking or "default",
            "max_tokens": max_tokens,
            "stream": stream,
            "content_empty": not bool((content or "").strip()),
        }
    )
if image_meta:
    meta.update(image_meta)
if error:
    meta["error"] = error
(out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

lines = [
    f"# {case_name}",
    "",
    f"- mode: `{mode}`",
    f"- model: `{model_name}`",
    f"- http_status: `{http_status}`",
    f"- elapsed_sec: `{elapsed}`",
    f"- verdict: `{verdict}`",
    f"- command: `{command}`",
]
if vram:
    lines.append(
        f"- VRAM before/peak/after MiB: `{vram.get('before_mib')}` / `{vram.get('peak_mib')}` / `{vram.get('after_mib')}` (total `{vram.get('total_mib')}`)"
    )
if mode == "chat":
    lines += [
        f"- thinking: `{thinking or 'default'}`",
        f"- max_tokens: `{max_tokens}`",
        f"- stream: `{stream}`",
        "",
        "## Prompt",
        "",
        prompt or "(empty)",
        "",
    ]
    if image_meta and image_meta.get("image_file"):
        lines += [
            f"image_file: `{image_meta['image_file']}` ({image_meta.get('image_bytes')} bytes)",
            "",
        ]
    if image_meta and image_meta.get("image_url"):
        lines += [f"image_url: `{image_meta['image_url']}`", ""]
    lines += [
        "## Content",
        "",
        content if content else "(empty)",
        "",
        "## Reasoning",
        "",
        reasoning if reasoning else "(empty)",
        "",
    ]
else:
    lines += ["", "## Request", "", f"`{method} {url}`", ""]
    if payload is not None:
        lines += ["```json", json.dumps(payload, indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Response", ""]
    if response_obj is not None:
        shown = json.loads(json.dumps(response_obj))
        if isinstance(shown, dict) and "raw_sse" in shown:
            shown["raw_sse"] = "<see response.sse>"
        lines += ["```json", json.dumps(shown, indent=2, ensure_ascii=False), "```", ""]
    else:
        lines += [response_text or "(empty)", ""]

if error:
    lines += ["## Error", "", error, ""]
if fail_reasons:
    lines += ["## Expect", ""]
    for reason in fail_reasons:
        lines += [f"- {reason}"]
    lines.append("")
if usage:
    lines += ["## Usage", "", "```json", json.dumps(usage, indent=2), "```", ""]
if timings:
    lines += ["## Timings", "", "```json", json.dumps(timings, indent=2), "```", ""]
(out / "summary.md").write_text("\n".join(lines) + "\n")

print(json.dumps({
    "out_dir": str(out),
    "http_status": http_status,
    "elapsed_sec": elapsed,
    "verdict": verdict,
    "output_summary": meta["output_summary"],
}, ensure_ascii=False))
if verdict != "PASS":
    raise SystemExit(1)
PY
