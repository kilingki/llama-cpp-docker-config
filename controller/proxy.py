import asyncio
import logging
from collections.abc import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from controller.lifecycle import Lifecycle

logger = logging.getLogger("controller.proxy")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

QueueItem = tuple


def _filtered_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP
    }


def _filtered_response_headers(headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP
    }


def _unavailable() -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "model is not ready"})


def _counts_as_inference(method: str, path: str) -> bool:
    return not (method.upper() == "GET" and path.strip("/") == "models")


async def open_upstream(
    client: httpx.AsyncClient,
    method: str,
    target: str,
    headers: dict[str, str],
    content: bytes | None,
    params,
) -> httpx.Response:
    upstream = client.build_request(
        method,
        target,
        headers=headers,
        content=content,
        params=params,
    )
    return await client.send(upstream, stream=True)


async def _drain_queue(queue: asyncio.Queue[QueueItem]) -> None:
    while True:
        item = await queue.get()
        if item[0] in {"end", "error"}:
            return


async def _continue_after_cancel(getter: asyncio.Task[QueueItem], queue: asyncio.Queue[QueueItem]) -> None:
    item = await getter
    if item[0] in {"end", "error"}:
        return
    await _drain_queue(queue)


async def proxy_openai(
    request: Request,
    path: str,
    lifecycle: Lifecycle,
    client: httpx.AsyncClient,
) -> JSONResponse | StreamingResponse:
    body = await request.body()
    counted = _counts_as_inference(request.method, path)
    if counted:
        if not await lifecycle.try_admit():
            return _unavailable()
    elif not await lifecycle.is_ready():
        return _unavailable()

    base = _lifecycle_base(lifecycle)
    target = f"{base}/v1/{path}" if path else f"{base}/v1"
    headers = _filtered_request_headers(request)
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()

    async def upstream() -> None:
        response = None
        try:
            response = await open_upstream(
                client,
                request.method,
                target,
                headers,
                body if body else None,
                request.query_params,
            )
            await queue.put(
                (
                    "head",
                    response.status_code,
                    _filtered_response_headers(response.headers),
                    response.headers.get("content-type"),
                )
            )
            async for chunk in response.aiter_raw():
                await queue.put(("chunk", chunk))
            await queue.put(("end",))
        except Exception as exc:
            await queue.put(("error", exc))
        finally:
            if response is not None:
                await response.aclose()
            if counted:
                await lifecycle.release()

    upstream_task = asyncio.create_task(upstream(), name="llama-proxy")
    getter = asyncio.create_task(queue.get())
    try:
        head = await asyncio.shield(getter)
    except asyncio.CancelledError:
        asyncio.create_task(_continue_after_cancel(getter, queue))
        raise

    if head[0] == "error":
        upstream_task.add_done_callback(_log_task_error)
        return _unavailable()

    _, status_code, response_headers, media_type = head

    async def stream() -> AsyncIterator[bytes]:
        finished = False
        try:
            while True:
                item = await queue.get()
                if item[0] == "chunk":
                    yield item[1]
                    continue
                finished = True
                return
        finally:
            if not finished:
                asyncio.create_task(_drain_queue(queue))

    return StreamingResponse(
        stream(),
        status_code=status_code,
        headers=response_headers,
        media_type=media_type,
    )


def _lifecycle_base(lifecycle: Lifecycle) -> str:
    if lifecycle.settings is None:
        return "http://127.0.0.1:8080"
    return lifecycle.settings.llama_base_url


def _log_task_error(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("upstream proxy task failed", exc_info=exc)
