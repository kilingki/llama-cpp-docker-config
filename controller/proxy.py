from collections.abc import AsyncIterator

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from controller.lifecycle import ProcessManager
from controller.schemas import ModelState

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


def _filtered_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP
    }


def _filtered_response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP
    }


async def proxy_openai(
    request: Request,
    path: str,
    manager: ProcessManager,
    client: httpx.AsyncClient,
) -> StreamingResponse:
    if manager.state != ModelState.READY or manager.pid is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    target = f"{manager.settings.llama_base_url}/v1/{path}" if path else f"{manager.settings.llama_base_url}/v1"
    body = await request.body()
    headers = _filtered_request_headers(request)

    try:
        upstream = client.build_request(
            request.method,
            target,
            headers=headers,
            content=body if body else None,
            params=request.query_params,
        )
        response = await client.send(upstream, stream=True)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Model is not loaded") from exc

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="Model is not loaded") from exc
        finally:
            await response.aclose()

    return StreamingResponse(
        stream(),
        status_code=response.status_code,
        headers=_filtered_response_headers(response),
        media_type=response.headers.get("content-type"),
    )
