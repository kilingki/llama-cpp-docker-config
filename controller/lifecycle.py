import asyncio
import logging
import os
import signal
from typing import Any

import httpx

from controller.config import Settings
from controller.schemas import ModelState

logger = logging.getLogger("controller.lifecycle")


class LifecycleError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


class ProcessManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = ModelState.UNLOADED
        self.proc: asyncio.subprocess.Process | None = None
        self.last_error: str | None = None
        self._lock = asyncio.Lock()
        self._log_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None

    @property
    def pid(self) -> int | None:
        if self.proc is None or self.proc.returncode is not None:
            return None
        return self.proc.pid

    def status_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "model": self.settings.model_name or None,
            "pid": self.pid,
            "backend": "llama.cpp",
            "runtime_port": self.settings.llama_port,
        }

    def llama_server_health_label(self) -> str:
        if self.state == ModelState.READY:
            return "ok" if self.pid is not None else "error"
        if self.state == ModelState.LOADING:
            return "loading"
        if self.state == ModelState.UNLOADING:
            return "stopping"
        if self.state == ModelState.ERROR:
            return "error"
        return "stopped"

    async def load(self, requested_model: str | None = None) -> dict[str, Any]:
        async with self._lock:
            if requested_model and requested_model != self.settings.model_name:
                raise LifecycleError(
                    "Model switching is not implemented in this version; "
                    f"configured model is '{self.settings.model_name}'",
                    status_code=400,
                )
            if self.state == ModelState.READY and self.pid is not None:
                return {
                    "status": "already_loaded",
                    "state": self.state,
                    "model": self.settings.model_name or None,
                    "pid": self.pid,
                }
            if self.state in {ModelState.LOADING, ModelState.UNLOADING}:
                raise LifecycleError(
                    f"Cannot load while state is {self.state.value}",
                    status_code=409,
                )
            self.state = ModelState.LOADING
            self.last_error = None

        await self._terminate_process()

        try:
            self._validate_model_files()
            await self._start_process()
            await self._wait_until_ready()
        except Exception as exc:
            await self._terminate_process()
            async with self._lock:
                self.state = ModelState.ERROR
                self.last_error = str(exc)
            logger.exception("Model load failed")
            raise LifecycleError(f"Model load failed: {exc}", status_code=500) from exc

        async with self._lock:
            if self.state != ModelState.LOADING:
                raise LifecycleError(
                    f"Load aborted; current state is {self.state.value}",
                    status_code=409,
                )
            self.state = ModelState.READY
            self._watch_task = asyncio.create_task(self._watch_process(), name="llama-watch")
            return {
                "status": "loaded",
                "state": self.state,
                "model": self.settings.model_name or None,
                "pid": self.pid,
            }

    async def unload(self) -> dict[str, Any]:
        async with self._lock:
            if self.state == ModelState.UNLOADED and self.pid is None:
                return {"status": "already_unloaded", "state": self.state}
            if self.state in {ModelState.LOADING, ModelState.UNLOADING}:
                raise LifecycleError(
                    f"Cannot unload while state is {self.state.value}",
                    status_code=409,
                )
            self.state = ModelState.UNLOADING

        try:
            await self._terminate_process()
        except Exception as exc:
            async with self._lock:
                self.state = ModelState.ERROR
                self.last_error = str(exc)
            logger.exception("Model unload failed")
            raise LifecycleError(f"Model unload failed: {exc}", status_code=500) from exc

        async with self._lock:
            self.state = ModelState.UNLOADED
            self.last_error = None
            return {"status": "unloaded", "state": self.state}

    async def shutdown(self) -> None:
        logger.info("Controller shutting down; stopping llama-server if running")
        try:
            await self._terminate_process()
        except Exception:
            logger.exception("Failed to stop llama-server during shutdown")
        async with self._lock:
            self.state = ModelState.UNLOADED

    def _validate_model_files(self) -> None:
        model_path = self.settings.model_path
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")
        mmproj_path = self.settings.mmproj_path
        if mmproj_path and not os.path.isfile(mmproj_path):
            raise FileNotFoundError(f"MMPROJ_PATH does not exist: {mmproj_path}")

    async def _start_process(self) -> None:
        cmd = self.settings.llama_server_cmd()
        logger.info("Starting llama-server: %s", " ".join(cmd))
        self.proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self._log_task = asyncio.create_task(self._pump_logs(), name="llama-logs")

    async def _wait_until_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.settings.load_timeout_sec
        url = f"{self.settings.llama_base_url}/health"
        async with httpx.AsyncClient() as client:
            while True:
                if self.proc is None:
                    raise RuntimeError("llama-server process handle is missing")
                if self.proc.returncode is not None:
                    raise RuntimeError(
                        f"llama-server exited during load with code {self.proc.returncode}"
                    )
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(
                        f"llama-server did not become ready within {self.settings.load_timeout_sec}s"
                    )
                try:
                    response = await client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        logger.info("llama-server health check passed")
                        return
                except httpx.RequestError:
                    pass
                await asyncio.sleep(0.5)

    async def _watch_process(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            returncode = await proc.wait()
        except asyncio.CancelledError:
            raise
        async with self._lock:
            if self.state == ModelState.READY:
                self.state = ModelState.ERROR
                self.last_error = f"llama-server exited unexpectedly with code {returncode}"
                logger.error(self.last_error)
                self.proc = None

    async def _terminate_process(self) -> None:
        async with self._lock:
            watch = self._watch_task
            self._watch_task = None
            proc = self.proc
            log_task = self._log_task
            self._log_task = None

        if watch is not None:
            watch.cancel()
            try:
                await watch
            except asyncio.CancelledError:
                pass

        if proc is None:
            if log_task is not None:
                await self._await_log_task(log_task)
            return

        if proc.returncode is None:
            pid = proc.pid
            logger.info("Sending SIGTERM to llama-server process group pid=%s", pid)
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.settings.unload_timeout_sec)
            except TimeoutError:
                logger.warning("llama-server did not exit after SIGTERM; sending SIGKILL")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except TimeoutError as exc:
                    raise RuntimeError("llama-server process group did not exit after SIGKILL") from exc

        async with self._lock:
            if self.proc is proc:
                self.proc = None

        if log_task is not None:
            await self._await_log_task(log_task)
        logger.info("llama-server process reaped")

    async def _await_log_task(self, task: asyncio.Task[None]) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=1)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _pump_logs(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
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
