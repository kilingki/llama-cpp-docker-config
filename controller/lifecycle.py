import asyncio
import logging
import os
import signal
from typing import Any, Protocol

import httpx

from controller.config import Settings

logger = logging.getLogger("controller.lifecycle")


def _retrieve_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    task.exception()


class ControlError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class WorkerFailure(Exception):
    def __init__(self, message: str, residency: str) -> None:
        super().__init__(message)
        self.residency = residency
        self.message = message


class ModelWorker(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def is_alive(self) -> bool: ...


def _group_alive(pid: int) -> bool | None:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


class LlamaServerWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._proc: asyncio.subprocess.Process | None = None
        self._pid: int | None = None
        self._log_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None

    def is_alive(self) -> bool:
        if self._proc is not None and self._proc.returncode is None:
            return True
        if self._pid is None:
            return False
        alive = _group_alive(self._pid)
        return alive is True

    def arm_watch(self, handler) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is not None:
            asyncio.create_task(handler(proc.returncode))
            return
        self._watch_task = asyncio.create_task(self._watch(handler), name="llama-watch")

    async def start(self) -> None:
        if self.is_alive():
            await self._wait_until_ready()
            return
        await self._reap_logs()
        self._validate_model_files()
        await self._spawn()
        try:
            await self._wait_until_ready()
        except WorkerFailure:
            if not self.is_alive():
                await self._reap_logs()
            raise

    async def stop(self) -> None:
        await self._cancel_watch()
        proc = self._proc
        pid = self._pid
        if not self.is_alive():
            self._proc = None
            self._pid = None
            await self._reap_logs()
            return
        self._signal_group(pid, proc, signal.SIGTERM)
        if await self._wait_dead(self.settings.unload_timeout_sec):
            self._proc = None
            self._pid = None
            await self._reap_logs()
            return
        logger.warning("llama-server did not exit after SIGTERM; sending SIGKILL")
        self._signal_group(pid, proc, signal.SIGKILL)
        if await self._wait_dead(5):
            self._proc = None
            self._pid = None
            await self._reap_logs()
            return
        residency = "resident" if self.is_alive() else "unknown"
        raise WorkerFailure("failed to release model", residency)

    def _validate_model_files(self) -> None:
        model_path = self.settings.model_path
        if not model_path or not os.path.isfile(model_path):
            raise WorkerFailure(f"MODEL_PATH does not exist: {model_path}", "not_resident")
        mmproj_path = self.settings.mmproj_path
        if mmproj_path and not os.path.isfile(mmproj_path):
            raise WorkerFailure(f"MMPROJ_PATH does not exist: {mmproj_path}", "not_resident")

    async def _spawn(self) -> None:
        try:
            cmd = self.settings.llama_server_cmd()
        except ValueError as exc:
            raise WorkerFailure(str(exc), "not_resident") from exc
        logger.info("Starting llama-server: %s", " ".join(cmd))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            self._proc = None
            self._pid = None
            raise WorkerFailure(str(exc), "not_resident") from exc
        self._pid = self._proc.pid
        self._log_task = asyncio.create_task(self._pump_logs(self._proc), name="llama-logs")

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.settings.load_timeout_sec
        url = f"{self.settings.llama_base_url}/health"
        async with httpx.AsyncClient() as client:
            while True:
                proc = self._proc
                if proc is not None and proc.returncode is not None:
                    raise WorkerFailure(
                        f"llama-server exited during load with code {proc.returncode}",
                        "not_resident",
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    residency = "unknown" if self.is_alive() else "not_resident"
                    raise WorkerFailure(
                        f"llama-server did not become ready within {self.settings.load_timeout_sec}s",
                        residency,
                    )
                try:
                    response = await client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        logger.info("llama-server health check passed")
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)

    async def _watch(self, handler) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            returncode = await proc.wait()
        except asyncio.CancelledError:
            raise
        await handler(returncode)

    async def _cancel_watch(self) -> None:
        watch = self._watch_task
        self._watch_task = None
        if watch is None:
            return
        watch.cancel()
        try:
            await watch
        except asyncio.CancelledError:
            pass

    def _signal_group(
        self,
        pid: int | None,
        proc: asyncio.subprocess.Process | None,
        sig: int,
    ) -> None:
        if pid is not None:
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                return
            except OSError as exc:
                raise WorkerFailure(f"failed to signal worker: {exc}", "unknown") from exc
            return
        if proc is None:
            return
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.send_signal(sig)

    async def _wait_dead(self, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            proc = self._proc
            if proc is not None and proc.returncode is None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return not self.is_alive()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=min(0.2, remaining))
                except TimeoutError:
                    pass
            if not self.is_alive():
                return True
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.2)

    async def _reap_logs(self) -> None:
        task = self._log_task
        self._log_task = None
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _pump_logs(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdout is None:
            return
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                logger.info("[llama-server] %s", line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed while forwarding llama-server logs")


class Lifecycle:
    def __init__(self, settings: Settings | None = None, worker: ModelWorker | None = None) -> None:
        if worker is None:
            if settings is None:
                raise ValueError("settings are required")
            worker = LlamaServerWorker(settings)
        self._worker = worker
        self.settings = settings
        self._lock = asyncio.Lock()
        self._state = "unloaded"
        self._residency = "not_resident"
        self._active = 0
        self._last_error: dict[str, str] | None = None
        self._operation: asyncio.Task[dict[str, Any]] | None = None
        self._status_fault: str | None = None

    def worker_alive(self) -> bool:
        return self._worker.is_alive()

    def fail_status(self, message: str = "status is unavailable") -> None:
        self._status_fault = message

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            if self._status_fault is not None:
                raise ControlError(500, "STATUS_FAILED", self._status_fault)
            return self._snapshot()

    async def is_ready(self) -> bool:
        async with self._lock:
            return self._state == "ready" and self._residency == "resident"

    async def load(self) -> dict[str, Any]:
        async with self._lock:
            if self._state == "ready" and self._worker.is_alive():
                return self._snapshot()
            if self._state == "loading" and self._operation is not None:
                task = self._operation
            elif self._state == "unloading":
                raise ControlError(409, "LIFECYCLE_CONFLICT", "unload is in progress")
            elif self._state == "failed" and self._residency != "not_resident":
                raise ControlError(
                    409,
                    "LIFECYCLE_CONFLICT",
                    "load is not allowed while residency remains",
                )
            else:
                self._state = "loading"
                task = asyncio.create_task(self._run_load(), name="load")
                task.add_done_callback(_retrieve_task_error)
                self._operation = task
        return await self._await_operation(task)

    async def unload(self) -> dict[str, Any]:
        async with self._lock:
            if self._state == "unloading" and self._operation is not None:
                task = self._operation
            elif self._active > 0:
                raise ControlError(409, "BUSY", "runtime has active inference requests")
            elif self._state == "loading":
                raise ControlError(409, "LIFECYCLE_CONFLICT", "load is in progress")
            elif self._state == "unloaded" and self._residency == "not_resident":
                return self._snapshot()
            elif self._state == "failed" and self._residency == "not_resident":
                self._state = "unloaded"
                self._residency = "not_resident"
                self._last_error = None
                return self._snapshot()
            else:
                self._state = "unloading"
                task = asyncio.create_task(self._run_unload(), name="unload")
                task.add_done_callback(_retrieve_task_error)
                self._operation = task
        return await self._await_operation(task)

    async def try_admit(self) -> bool:
        async with self._lock:
            if self._state != "ready" or self._residency != "resident":
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active > 0:
                self._active -= 1

    async def note_worker_exit(self, returncode: int) -> None:
        async with self._lock:
            if self._state != "ready":
                return
            self._state = "failed"
            self._residency = "resident" if self._worker.is_alive() else "not_resident"
            self._last_error = {
                "code": "WORKER_EXITED",
                "message": f"llama-server exited unexpectedly with code {returncode}",
            }
            logger.error(self._last_error["message"])

    async def shutdown(self) -> None:
        logger.info("Controller shutting down; stopping llama-server if running")
        if self.worker_alive():
            try:
                await self._worker.stop()
            except WorkerFailure:
                logger.exception("Failed to stop llama-server during shutdown")
                return
        async with self._lock:
            self._state = "unloaded"
            self._residency = "not_resident"
            self._active = 0
            self._last_error = None

    async def _await_operation(self, task: asyncio.Task[dict[str, Any]]) -> dict[str, Any]:
        try:
            return await asyncio.shield(task)
        except WorkerFailure as exc:
            code = getattr(exc, "control_code", "LOAD_FAILED")
            raise ControlError(500, code, exc.message) from exc

    async def _run_load(self) -> dict[str, Any]:
        try:
            await self._worker.start()
        except WorkerFailure as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = exc.residency
                self._last_error = {"code": "LOAD_FAILED", "message": exc.message}
                self._operation = None
            exc.control_code = "LOAD_FAILED"  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = "unknown"
                self._last_error = {"code": "LOAD_FAILED", "message": str(exc)}
                self._operation = None
            failure = WorkerFailure(str(exc), "unknown")
            failure.control_code = "LOAD_FAILED"  # type: ignore[attr-defined]
            raise failure from exc
        async with self._lock:
            if not self._worker.is_alive():
                self._state = "failed"
                self._residency = "not_resident"
                self._last_error = {
                    "code": "LOAD_FAILED",
                    "message": "llama-server exited during load",
                }
                self._operation = None
                failure = WorkerFailure("llama-server exited during load", "not_resident")
                failure.control_code = "LOAD_FAILED"  # type: ignore[attr-defined]
                raise failure
            self._state = "ready"
            self._residency = "resident"
            self._last_error = None
            self._operation = None
            snapshot = self._snapshot()
        arm = getattr(self._worker, "arm_watch", None)
        if arm is not None:
            arm(self.note_worker_exit)
        return snapshot

    async def _run_unload(self) -> dict[str, Any]:
        try:
            await self._worker.stop()
        except WorkerFailure as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = exc.residency
                self._last_error = {"code": "UNLOAD_FAILED", "message": exc.message}
                self._operation = None
            exc.control_code = "UNLOAD_FAILED"  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            async with self._lock:
                try:
                    alive = self._worker.is_alive()
                except Exception:
                    residency = "unknown"
                else:
                    residency = "resident" if alive else "unknown"
                self._state = "failed"
                self._residency = residency
                self._last_error = {"code": "UNLOAD_FAILED", "message": str(exc)}
                self._operation = None
            failure = WorkerFailure(str(exc), residency)
            failure.control_code = "UNLOAD_FAILED"  # type: ignore[attr-defined]
            raise failure from exc
        async with self._lock:
            self._state = "unloaded"
            self._residency = "not_resident"
            self._active = 0
            self._last_error = None
            self._operation = None
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        last_error = None if self._last_error is None else dict(self._last_error)
        return {
            "state": self._state,
            "residency": self._residency,
            "active_requests": self._active,
            "last_error": last_error,
        }
