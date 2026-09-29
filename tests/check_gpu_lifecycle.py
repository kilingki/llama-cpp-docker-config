#!/usr/bin/env python3
"""Measure load, chat, and unload against the running llama.cpp container.

Records host GPU memory and whether the llama-server process is gone after
unload. Measured MiB values are observations, not resourceProfile byte budgets.
"""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
STATUS_LIMIT_SEC = 2.0
ROUNDS = 3
RESIDUAL_LIMIT_MIB = 256
OUTPUT = Path(__file__).resolve().parent / "outputs" / "gpu_lifecycle.md"


def gpu_mib() -> int:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(raw.strip().splitlines()[0])


def container_id() -> str:
    raw = subprocess.check_output(
        ["docker", "compose", "ps", "-q", "llama-runtime"],
        cwd=ROOT,
        text=True,
    ).strip()
    if not raw:
        raise RuntimeError("llama-runtime container is not running")
    return raw.splitlines()[0]


def worker_cmds(cid: str) -> list[str]:
    # The search string stays in the environment. Putting it in this shell's
    # command line makes /proc/*/cmdline match the scanner itself.
    script = r"""
self=$$
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  [ "$pid" = "$self" ] && continue
  cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null || true)
  case "$cmd" in
    *"$NEEDLE"*) printf '%s\n' "$cmd" ;;
  esac
done
"""
    raw = subprocess.check_output(
        ["docker", "exec", "-e", "NEEDLE=llama-server", cid, "sh", "-c", script],
        text=True,
    )
    return [line for line in raw.splitlines() if line.strip()]


def request(method: str, path: str, body: bytes | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = body if body not in (None, b"") else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, json.loads(raw) if raw else {}


def chat() -> tuple[int, dict]:
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    return request("POST", "/v1/chat/completions", payload)


def watch_status(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        started = time.perf_counter()
        try:
            request("GET", "/control/status", timeout=STATUS_LIMIT_SEC)
        except Exception:
            pass
        samples.append(time.perf_counter() - started)
        time.sleep(0.05)


def watch_memory(stop: threading.Event, peaks: list[int]) -> None:
    peak = gpu_mib()
    while not stop.is_set():
        peak = max(peak, gpu_mib())
        time.sleep(0.05)
    peaks.append(peak)


def write_report(rounds: list[dict], summary: dict | None, error: str | None) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GPU lifecycle 테스트 결과",
        "",
        "llama-server 자식 프로세스에 대해 load, chat, unload를 반복한 기록이다.",
        "메모리는 호스트 nvidia-smi 사용량이다. 측정하지 않은 byte budget은 적지 않는다.",
        "",
    ]
    if error:
        lines.extend([f"실패: {error}", ""])
    for item in rounds:
        lines.extend(
            [
                f"## round {item['round']} {item['step']}",
                "",
                "```json",
                json.dumps(item, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    if summary is not None:
        lines.extend(["## summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def fail(rounds: list[dict], message: str) -> None:
    write_report(rounds, None, message)
    raise SystemExit(message)


def main() -> None:
    try:
        status, initial = request("GET", "/control/status", timeout=5)
    except Exception as exc:
        fail([], f"control status is not reachable: {exc}")
    if status != 200 or initial.get("state") != "unloaded" or initial.get("residency") != "not_resident":
        fail(
            [{"round": 0, "step": "startup", "http_status": status, "body": initial}],
            "startup is not unloaded",
        )

    cid = container_id()
    rounds: list[dict] = []
    load_peaks: list[int] = []
    infer_peaks: list[int] = []
    residuals: list[int] = []

    for round_index in range(ROUNDS):
        before = gpu_mib()
        stop = threading.Event()
        latencies: list[float] = []
        peaks: list[int] = []
        threading.Thread(target=watch_status, args=(stop, latencies), daemon=True).start()
        threading.Thread(target=watch_memory, args=(stop, peaks), daemon=True).start()
        started = time.perf_counter()
        load_status, load_body = request("POST", "/control/load", b"{}")
        stop.set()
        time.sleep(0.2)
        load_sec = time.perf_counter() - started
        worst = max(latencies) if latencies else STATUS_LIMIT_SEC
        load_peak = max(peaks) if peaks else gpu_mib()
        rounds.append(
            {
                "round": round_index,
                "step": "load",
                "http_status": load_status,
                "body": load_body,
                "seconds": round(load_sec, 3),
                "status_max_sec": round(worst, 3),
                "mib_before": before,
                "mib_after": gpu_mib(),
                "load_peak_mib": load_peak,
            }
        )
        if load_status != 200 or load_body.get("state") != "ready" or worst >= STATUS_LIMIT_SEC:
            fail(rounds, "load did not keep status responsive")
        load_peaks.append(load_peak)

        stop = threading.Event()
        infer_latencies: list[float] = []
        infer_peak_box: list[int] = []
        threading.Thread(target=watch_status, args=(stop, infer_latencies), daemon=True).start()
        threading.Thread(target=watch_memory, args=(stop, infer_peak_box), daemon=True).start()
        chat_status, chat_body = chat()
        stop.set()
        time.sleep(0.2)
        infer_worst = max(infer_latencies) if infer_latencies else STATUS_LIMIT_SEC
        infer_peak = max(infer_peak_box) if infer_peak_box else gpu_mib()
        rounds.append(
            {
                "round": round_index,
                "step": "chat",
                "http_status": chat_status,
                "text": str(chat_body)[:500],
                "status_max_sec": round(infer_worst, 3),
                "infer_peak_mib": infer_peak,
            }
        )
        if chat_status != 200 or infer_worst >= STATUS_LIMIT_SEC:
            fail(rounds, "inference did not keep status responsive")
        infer_peaks.append(infer_peak)

        unload_status, unload_body = request("POST", "/control/unload")
        time.sleep(2)
        workers = worker_cmds(cid)
        residual = gpu_mib()
        residuals.append(residual)
        rounds.append(
            {
                "round": round_index,
                "step": "unload",
                "http_status": unload_status,
                "body": unload_body,
                "worker_cmds": workers,
                "mib_after": residual,
            }
        )
        if (
            unload_status != 200
            or unload_body.get("state") != "unloaded"
            or unload_body.get("residency") != "not_resident"
            or workers
        ):
            fail(rounds, "unload did not release the worker")

    grew = residuals[-1] > residuals[0] + RESIDUAL_LIMIT_MIB
    summary = {
        "load_peak_mib": max(load_peaks),
        "infer_peak_mib": max(infer_peaks),
        "residuals_mib": residuals,
        "residual_grew": grew,
    }
    write_report(rounds, summary, "GPU memory grew across unload rounds" if grew else None)
    if grew:
        raise SystemExit("GPU memory grew across unload rounds")


if __name__ == "__main__":
    main()
