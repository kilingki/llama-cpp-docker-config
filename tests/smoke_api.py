#!/usr/bin/env python3
"""API smoke against a running container.

Sequence, after forcing an unloaded model:
health 200 with llama_server stopped, /v1/models 503, load, chat pong,
streaming chat pong, unload, reload. The last step leaves the model loaded.

Image description and reasoning-effort cases live in tests/infer_qwen.py.
GPU residual rounds live in tests/check_gpu_lifecycle.py.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.8-27b")
REQUEST_TIMEOUT_SEC = float(os.environ.get("REQUEST_TIMEOUT_SEC", "600"))
PROMPT = "Reply with the single word: pong"
EXPECT = "pong"


def request_raw(
    method: str,
    path: str,
    body: dict | None = None,
    timeout: float = REQUEST_TIMEOUT_SEC,
) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def request_json(method: str, path: str, body: dict | None = None, timeout: float = REQUEST_TIMEOUT_SEC) -> tuple[int, dict]:
    status, raw = request_raw(method, path, body, timeout)
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text else {}
    except json.JSONDecodeError:
        parsed = {"raw": text}
    if not isinstance(parsed, dict):
        parsed = {"raw": parsed}
    return status, parsed


def parse_sse(raw_text: str) -> dict:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
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
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = (choices[0] or {}).get("delta") or {}
        piece = delta.get("content")
        if isinstance(piece, str):
            content_parts.append(piece)
        thought = delta.get("reasoning_content")
        if isinstance(thought, str):
            reasoning_parts.append(thought)
    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
    }


def chat_body(*, stream: bool) -> dict:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def message_content(body: dict) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def step_record(name: str, http_status: int, body: dict, elapsed: float, ok: bool, detail: str) -> dict:
    return {
        "name": name,
        "http_status": http_status,
        "e2e_sec": round(elapsed, 3),
        "ok": ok,
        "detail": detail,
        "body": body,
    }


def timed_json(name: str, method: str, path: str, body: dict | None, ok_fn) -> dict:
    started = time.perf_counter()
    status, payload = request_json(method, path, body)
    elapsed = time.perf_counter() - started
    ok, detail = ok_fn(status, payload)
    return step_record(name, status, payload, elapsed, ok, detail)


def ensure_unloaded() -> dict | None:
    status, body = request_json("GET", "/control/status", timeout=10)
    if status == 200 and body.get("state") == "unloaded" and body.get("residency") == "not_resident":
        return None
    started = time.perf_counter()
    unload_status, unload_body = request_json("POST", "/control/unload", {})
    elapsed = time.perf_counter() - started
    ok = (
        unload_status == 200
        and unload_body.get("state") == "unloaded"
        and unload_body.get("residency") == "not_resident"
    )
    record = step_record(
        "00-prepare-unload",
        unload_status,
        {"status_before": body, "unload": unload_body},
        elapsed,
        ok,
        "unloaded" if ok else "unload did not reach not_resident",
    )
    if not ok:
        raise SystemExit(f"prepare unload failed: HTTP {unload_status} {unload_body}")
    return record


def write_index(run_dir: Path, rows: list[dict]) -> None:
    lines = [
        "# API 스모크",
        "",
        f"- base: `{BASE}`",
        f"- model: `{MODEL_NAME}`",
        f"- chat prompt: {PROMPT}",
        "",
        "언로드 상태에서 `/health`와 `/v1/models`를 확인하고, 채팅·스트림에서 `pong`을 본 뒤 언로드하고 다시 로드한다.",
        "",
        "| step | HTTP | e2e_s | ok | detail |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['http_status']} | {row['e2e_sec']:.3f} | {row['ok']} | {row['detail']} |"
        )
    lines.append("")
    (run_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(run_dir: Path, row: dict) -> None:
    (run_dir / f"{row['name']}.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).resolve().parent / "outputs" / f"{stamp}-smoke-api"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"results: {run_dir}", flush=True)

    rows: list[dict] = []
    prepared = ensure_unloaded()
    if prepared is not None:
        rows.append(prepared)
        save(run_dir, prepared)
        write_index(run_dir, rows)
        print(f"== {prepared['name']} {prepared['detail']}", flush=True)

    def health_ok(status: int, body: dict) -> tuple[bool, str]:
        ok = status == 200 and body.get("controller") == "ok" and body.get("llama_server") == "stopped"
        return ok, f"llama_server={body.get('llama_server')}"

    def models_ok(status: int, body: dict) -> tuple[bool, str]:
        return status == 503, str(body.get("detail") or body.get("error") or "")

    def ready_ok(status: int, body: dict) -> tuple[bool, str]:
        ok = status == 200 and body.get("state") == "ready" and body.get("residency") == "resident"
        return ok, f"state={body.get('state')} residency={body.get('residency')}"

    def unloaded_ok(status: int, body: dict) -> tuple[bool, str]:
        ok = status == 200 and body.get("state") == "unloaded" and body.get("residency") == "not_resident"
        return ok, f"state={body.get('state')} residency={body.get('residency')}"

    def chat_ok(status: int, body: dict) -> tuple[bool, str]:
        content = message_content(body)
        ok = status == 200 and EXPECT in content
        return ok, content or "(empty)"

    def stream_row() -> dict:
        started = time.perf_counter()
        status, raw = request_raw("POST", "/v1/chat/completions", chat_body(stream=True))
        elapsed = time.perf_counter() - started
        parsed = parse_sse(raw.decode("utf-8", errors="replace"))
        content = parsed["content"]
        ok = status == 200 and EXPECT in content
        return step_record(
            "05-chat-stream",
            status,
            {"content": content, "reasoning_content": parsed["reasoning_content"]},
            elapsed,
            ok,
            content or "(empty)",
        )

    failed = False
    json_steps = [
        ("01-health-unloaded", "GET", "/health", None, health_ok),
        ("02-models-expect-503", "GET", "/v1/models", None, models_ok),
        ("03-load", "POST", "/control/load", {}, ready_ok),
        ("04-chat", "POST", "/v1/chat/completions", chat_body(stream=False), chat_ok),
    ]
    for name, method, path, body, ok_fn in json_steps:
        print(f"== {name}", flush=True)
        row = timed_json(name, method, path, body, ok_fn)
        rows.append(row)
        save(run_dir, row)
        write_index(run_dir, rows)
        print(f"   HTTP {row['http_status']} e2e {row['e2e_sec']}s ok {row['ok']} {row['detail']}", flush=True)
        failed = failed or not row["ok"]

    print("== 05-chat-stream", flush=True)
    streamed = stream_row()
    rows.append(streamed)
    save(run_dir, streamed)
    write_index(run_dir, rows)
    print(
        f"   HTTP {streamed['http_status']} e2e {streamed['e2e_sec']}s ok {streamed['ok']} {streamed['detail']}",
        flush=True,
    )
    failed = failed or not streamed["ok"]

    for name, method, path, body, ok_fn in (
        ("06-unload", "POST", "/control/unload", {}, unloaded_ok),
        ("07-reload", "POST", "/control/load", {}, ready_ok),
    ):
        print(f"== {name}", flush=True)
        row = timed_json(name, method, path, body, ok_fn)
        rows.append(row)
        save(run_dir, row)
        write_index(run_dir, rows)
        print(f"   HTTP {row['http_status']} e2e {row['e2e_sec']}s ok {row['ok']} {row['detail']}", flush=True)
        failed = failed or not row["ok"]

    print(f"results: {run_dir / 'INDEX.md'}", flush=True)
    if failed:
        raise SystemExit("one or more smoke steps failed")


if __name__ == "__main__":
    main()
