import asyncio
import os
import sys
import unittest
import unittest.mock
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from fastapi.testclient import TestClient

from controller import main as api
from controller import proxy as proxy_mod
from controller.lifecycle import ControlError, Lifecycle, WorkerFailure


class GateWorker:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release_start = asyncio.Event()
        self.stop_entered = asyncio.Event()
        self.release_stop = asyncio.Event()
        self.alive = False
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        self.started.set()
        await self.release_start.wait()
        self.alive = True

    async def stop(self) -> None:
        self.stops += 1
        self.stop_entered.set()
        await self.release_stop.wait()
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class ScriptedWorker:
    def __init__(
        self,
        *,
        start_failure: str | None = None,
        stop_failure: str | None = None,
    ) -> None:
        self.start_failure = start_failure
        self.stop_failure = stop_failure
        self.alive = False
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        if self.start_failure is not None:
            self.alive = self.start_failure != "not_resident"
            raise WorkerFailure("load failed", self.start_failure)
        self.alive = True

    async def stop(self) -> None:
        self.stops += 1
        if self.stop_failure is not None:
            self.alive = self.stop_failure == "resident"
            raise WorkerFailure("failed to release model", self.stop_failure)
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class HoldingResponse:
    def __init__(self, gate: asyncio.Event, entered: asyncio.Event, seen: list[int]) -> None:
        self.status_code = 200
        self.headers = {"content-type": "text/plain"}
        self._gate = gate
        self._entered = entered
        self._seen = seen
        self._between = asyncio.Event()
        self.second = asyncio.Event()
        self.active_between: list[int] = []

    async def aiter_raw(self):
        self._seen[0] += 1
        self._entered.set()
        await self._gate.wait()
        yield b"a"
        self._between.set()
        await self.second.wait()
        yield b"b"

    async def aclose(self) -> None:
        return None


def _run(coro):
    return asyncio.run(coro)


class ControlLifecycleTest(unittest.TestCase):
    def test_initial_status_and_inference_unavailable(self) -> None:
        api.lifecycle = Lifecycle(worker=ScriptedWorker())
        with TestClient(api.app) as client:
            status = client.get("/control/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(
                status.json(),
                {
                    "state": "unloaded",
                    "residency": "not_resident",
                    "active_requests": 0,
                    "last_error": None,
                },
            )
            rejected = client.post("/v1/chat/completions", json={"messages": []})
            self.assertEqual(rejected.status_code, 503, rejected.text)
            self.assertEqual(rejected.json(), {"detail": "model is not ready"})
            after = client.get("/control/status").json()
            self.assertEqual(after["active_requests"], 0)
            bad = client.post("/control/load", json={"model": "qwen3.8-27b"})
            self.assertEqual(bad.status_code, 400)
            self.assertEqual(bad.json()["error"]["code"], "BAD_REQUEST")
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["llama_server"], "stopped")

    def test_load_unload_noop_and_models_proxy(self) -> None:
        worker = ScriptedWorker()
        api.lifecycle = Lifecycle(worker=worker)
        calls = {"n": 0}

        async def open_upstream(*_args, **_kwargs):
            calls["n"] += 1

            class Models:
                status_code = 200
                headers = {"content-type": "application/json"}

                async def aiter_raw(self):
                    yield b'{"data":[]}'

                async def aclose(self) -> None:
                    return None

            return Models()

        with TestClient(api.app) as client:
            missing = client.get("/v1/models")
            self.assertEqual(missing.status_code, 503)
            self.assertEqual(client.get("/control/status").json()["active_requests"], 0)
            loaded = client.post("/control/load", content=b"{}")
            self.assertEqual(loaded.status_code, 200, loaded.text)
            self.assertEqual(loaded.json()["state"], "ready")
            self.assertEqual(loaded.json()["residency"], "resident")
            self.assertIsNone(loaded.json()["last_error"])
            again = client.post("/control/load")
            self.assertEqual(again.status_code, 200, again.text)
            self.assertEqual(again.json()["state"], "ready")
            self.assertEqual(worker.starts, 1)
            with unittest.mock.patch.object(proxy_mod, "open_upstream", open_upstream):
                listed = client.get("/v1/models")
                self.assertEqual(listed.status_code, 200, listed.text)
                self.assertEqual(listed.content, b'{"data":[]}')
            self.assertEqual(calls["n"], 1)
            self.assertEqual(client.get("/control/status").json()["active_requests"], 0)
            unloaded = client.post("/control/unload")
            self.assertEqual(unloaded.status_code, 200, unloaded.text)
            self.assertEqual(
                unloaded.json(),
                {
                    "state": "unloaded",
                    "residency": "not_resident",
                    "active_requests": 0,
                    "last_error": None,
                },
            )
            repeated = client.post("/control/unload")
            self.assertEqual(repeated.status_code, 200, repeated.text)
            self.assertEqual(repeated.json()["state"], "unloaded")
            self.assertEqual(worker.stops, 1)

    def test_shared_load_and_conflict(self) -> None:
        async def scenario() -> None:
            import httpx

            worker = GateWorker()
            api.lifecycle = Lifecycle(worker=worker)
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first_task = asyncio.create_task(client.post("/control/load"))
                await worker.started.wait()
                status = await client.get("/control/status")
                conflict = await client.post("/control/unload")
                second_task = asyncio.create_task(client.post("/control/load"))
                worker.release_start.set()
                first, second = await asyncio.gather(first_task, second_task)
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["state"], "loading")
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.json(), second.json())
            self.assertEqual(first.json()["state"], "ready")
            self.assertEqual(worker.starts, 1)

        _run(scenario())

    def test_shared_unload_and_conflict(self) -> None:
        async def scenario() -> None:
            import httpx

            worker = GateWorker()
            api.lifecycle = Lifecycle(worker=worker)
            worker.release_start.set()
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                loaded = await client.post("/control/load")
                self.assertEqual(loaded.status_code, 200)
                first_task = asyncio.create_task(client.post("/control/unload"))
                await worker.stop_entered.wait()
                status = await client.get("/control/status")
                conflict = await client.post("/control/load")
                denied = await client.post("/v1/chat/completions", json={"messages": []})
                second_task = asyncio.create_task(client.post("/control/unload"))
                worker.release_stop.set()
                first, second = await asyncio.gather(first_task, second_task)
            self.assertEqual(status.json()["state"], "unloading")
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            self.assertEqual(denied.status_code, 503)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json(), second.json())
            self.assertEqual(first.json()["state"], "unloaded")
            self.assertEqual(first.json()["residency"], "not_resident")
            self.assertEqual(worker.stops, 1)

        _run(scenario())

    def test_disconnect_does_not_cancel_operation(self) -> None:
        async def scenario() -> None:
            import httpx

            worker = GateWorker()
            api.lifecycle = Lifecycle(worker=worker)
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                load_task = asyncio.create_task(client.post("/control/load"))
                await worker.started.wait()
                load_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await load_task
                loading = await client.get("/control/status")
                self.assertEqual(loading.status_code, 200)
                self.assertEqual(loading.json()["state"], "loading")
                worker.release_start.set()
                ready = None
                for _ in range(50):
                    ready = await client.get("/control/status")
                    if ready.json()["state"] == "ready":
                        break
                    await asyncio.sleep(0.01)
                assert ready is not None
                self.assertEqual(ready.json()["state"], "ready")
                self.assertEqual(worker.starts, 1)

                unload_task = asyncio.create_task(client.post("/control/unload"))
                await worker.stop_entered.wait()
                unload_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await unload_task
                unloading = await client.get("/control/status")
                self.assertEqual(unloading.json()["state"], "unloading")
                worker.release_stop.set()
                done = None
                for _ in range(50):
                    done = await client.get("/control/status")
                    if done.json()["state"] == "unloaded":
                        break
                    await asyncio.sleep(0.01)
            assert done is not None
            self.assertEqual(done.json()["state"], "unloaded")
            self.assertEqual(done.json()["residency"], "not_resident")
            self.assertEqual(worker.stops, 1)

        _run(scenario())

    def test_inference_accounting_and_busy(self) -> None:
        async def scenario() -> None:
            import httpx

            api.lifecycle = Lifecycle(worker=ScriptedWorker())
            entered = asyncio.Event()
            release = asyncio.Event()
            seen = [0]
            holders: list[HoldingResponse] = []

            async def open_upstream(*_args, **_kwargs):
                response = HoldingResponse(release, entered, seen)
                holders.append(response)
                return response

            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                loaded = await client.post("/control/load")
                self.assertEqual(loaded.status_code, 200)
                with unittest.mock.patch.object(proxy_mod, "open_upstream", open_upstream):
                    call = asyncio.create_task(
                        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
                    )
                    await entered.wait()
                    mid = await client.get("/control/status")
                    self.assertEqual(mid.json()["state"], "ready")
                    self.assertEqual(mid.json()["active_requests"], 1)
                    busy = await client.post("/control/unload")
                    self.assertEqual(busy.status_code, 409)
                    self.assertEqual(busy.json()["error"]["code"], "BUSY")
                    release.set()
                    await holders[0]._between.wait()
                    between = await client.get("/control/status")
                    self.assertEqual(between.json()["active_requests"], 1)
                    holders[0].second.set()
                    response = await call
                done = await client.get("/control/status")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.content, b"ab")
            self.assertEqual(done.json()["active_requests"], 0)
            unloaded = None
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                unloaded = await client.post("/control/unload")
            self.assertEqual(unloaded.status_code, 200)
            self.assertEqual(unloaded.json()["state"], "unloaded")

        _run(scenario())

    def test_two_requests_count_separately_and_cancel_keeps_active(self) -> None:
        async def scenario() -> None:
            import httpx

            api.lifecycle = Lifecycle(worker=ScriptedWorker())
            release = asyncio.Event()
            finish = asyncio.Event()
            both = asyncio.Event()
            seen = [0]

            class Response:
                def __init__(self) -> None:
                    self.status_code = 200
                    self.headers = {"content-type": "text/plain"}

                async def aiter_raw(self):
                    seen[0] += 1
                    if seen[0] >= 2:
                        both.set()
                    await release.wait()
                    yield b"a"
                    await finish.wait()
                    yield b"b"

                async def aclose(self) -> None:
                    return None

            async def open_upstream(*_args, **_kwargs):
                return Response()

            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                self.assertEqual((await client.post("/control/load")).status_code, 200)
                with unittest.mock.patch.object(proxy_mod, "open_upstream", open_upstream):
                    first = asyncio.create_task(client.post("/v1/completions", json={"prompt": "a"}))
                    second = asyncio.create_task(client.post("/v1/embeddings", json={"input": "b"}))
                    await asyncio.wait_for(both.wait(), timeout=5)
                    self.assertEqual((await client.get("/control/status")).json()["active_requests"], 2)
                    first.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await first
                    await asyncio.sleep(0.05)
                    self.assertEqual((await client.get("/control/status")).json()["active_requests"], 2)
                    release.set()
                    await asyncio.sleep(0.05)
                    self.assertEqual((await client.get("/control/status")).json()["active_requests"], 2)
                    finish.set()
                    second_response = await second
                    for _ in range(50):
                        if (await client.get("/control/status")).json()["active_requests"] == 0:
                            break
                        await asyncio.sleep(0.01)
                    done = await client.get("/control/status")
            self.assertEqual(second_response.status_code, 200)
            self.assertEqual(second_response.content, b"ab")
            self.assertEqual(done.json()["active_requests"], 0)

        _run(scenario())

    def test_load_failure_not_resident_can_retry(self) -> None:
        worker = ScriptedWorker(start_failure="not_resident")
        api.lifecycle = Lifecycle(worker=worker)
        with TestClient(api.app) as client:
            failed = client.post("/control/load")
            self.assertEqual(failed.status_code, 500)
            self.assertEqual(failed.json()["error"]["code"], "LOAD_FAILED")
            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["residency"], "not_resident")
            self.assertEqual(status["last_error"]["code"], "LOAD_FAILED")
            self.assertEqual(worker.stops, 0)
            worker.start_failure = None
            retried = client.post("/control/load")
            self.assertEqual(retried.status_code, 200, retried.text)
            self.assertEqual(retried.json()["state"], "ready")
            self.assertIsNone(retried.json()["last_error"])
            self.assertEqual(client.post("/control/unload").status_code, 200)
            worker.start_failure = "not_resident"
            failed_again = client.post("/control/load")
            self.assertEqual(failed_again.status_code, 500)
            stops_before = worker.stops
            noop = client.post("/control/unload")
            self.assertEqual(noop.status_code, 200, noop.text)
            self.assertEqual(noop.json()["state"], "unloaded")
            self.assertIsNone(noop.json()["last_error"])
            self.assertEqual(worker.stops, stops_before)

    def test_load_failure_unknown_blocks_load_until_recovery(self) -> None:
        worker = ScriptedWorker(start_failure="unknown")
        api.lifecycle = Lifecycle(worker=worker)
        with TestClient(api.app) as client:
            failed = client.post("/control/load")
            self.assertEqual(failed.status_code, 500)
            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["residency"], "unknown")
            blocked = client.post("/control/load")
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            recovered = client.post("/control/unload")
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json()["state"], "unloaded")
            self.assertEqual(recovered.json()["residency"], "not_resident")
            self.assertIsNone(recovered.json()["last_error"])

    def test_unload_failure_keeps_residency_and_recovery_clears_error(self) -> None:
        worker = ScriptedWorker(stop_failure="resident")
        api.lifecycle = Lifecycle(worker=worker)
        with TestClient(api.app) as client:
            self.assertEqual(client.post("/control/load").status_code, 200)
            failed = client.post("/control/unload")
            self.assertEqual(failed.status_code, 500)
            self.assertEqual(failed.json()["error"]["code"], "UNLOAD_FAILED")
            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["residency"], "resident")
            blocked = client.post("/control/load")
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            worker.stop_failure = None
            recovered = client.post("/control/unload")
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json()["state"], "unloaded")
            self.assertEqual(recovered.json()["residency"], "not_resident")
            self.assertIsNone(recovered.json()["last_error"])

    def test_status_failure_is_http_500(self) -> None:
        api.lifecycle = Lifecycle(worker=ScriptedWorker())
        api.lifecycle.fail_status()
        with TestClient(api.app) as client:
            status = client.get("/control/status")
            self.assertEqual(status.status_code, 500)
            self.assertEqual(status.json()["error"]["code"], "STATUS_FAILED")
            self.assertNotIn("state", status.json())

    def test_admit_and_unload_are_ordered(self) -> None:
        async def scenario() -> None:
            worker = GateWorker()
            life = Lifecycle(worker=worker)
            worker.release_start.set()
            await life.load()
            self.assertTrue(await life.try_admit())
            with self.assertRaises(ControlError) as caught:
                await life.unload()
            self.assertEqual(caught.exception.code, "BUSY")
            self.assertEqual(worker.stops, 0)
            await life.release()

            unload_task = asyncio.create_task(life.unload())
            await worker.stop_entered.wait()
            self.assertFalse(await life.try_admit())
            self.assertEqual((await life.status())["state"], "unloading")
            worker.release_stop.set()
            body = await unload_task
            self.assertEqual(body["state"], "unloaded")

        _run(scenario())

    def test_worker_exit_marks_failed(self) -> None:
        async def scenario() -> None:
            worker = ScriptedWorker()
            life = Lifecycle(worker=worker)
            await life.load()
            await life.note_worker_exit(1)
            still_resident = await life.status()
            self.assertEqual(still_resident["state"], "failed")
            self.assertEqual(still_resident["residency"], "resident")
            self.assertEqual(still_resident["last_error"]["code"], "WORKER_EXITED")
            with self.assertRaises(ControlError) as blocked:
                await life.load()
            self.assertEqual(blocked.exception.code, "LIFECYCLE_CONFLICT")
            worker.alive = False
            recovered = await life.unload()
            self.assertEqual(recovered["state"], "unloaded")
            self.assertEqual(recovered["residency"], "not_resident")
            self.assertIsNone(recovered["last_error"])

        _run(scenario())


class PrepareScriptTest(unittest.TestCase):
    def test_prepare_script_preserves_existing_runtime(self) -> None:
        path = _REPO_ROOT / "prepare-inferswap"
        text = path.read_text(encoding="utf-8")
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn("docker compose up -d", text)
        self.assertIn("not_resident", text)
        self.assertIn("not restarting", text)


if __name__ == "__main__":
    unittest.main()
