import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from controller.config import Settings
from controller.lifecycle import ControlError, Lifecycle
from controller.proxy import proxy_openai
from controller.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("controller")

settings = Settings.from_env()
lifecycle = Lifecycle(settings)
http_client = httpx.AsyncClient(timeout=httpx.Timeout(None))


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Controller started; model=%s path=%s (not loaded until POST /control/load)",
        settings.model_name,
        settings.model_path,
    )
    try:
        yield
    finally:
        await lifecycle.shutdown()
        await http_client.aclose()


app = FastAPI(title="llama.cpp runtime controller", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        controller="ok",
        llama_server="ok" if lifecycle.worker_alive() else "stopped",
    )


@app.get("/control/status")
async def control_status() -> JSONResponse:
    try:
        body = await lifecycle.status()
    except ControlError as exc:
        return _control_error(exc)
    return JSONResponse(body)


@app.post("/control/load")
async def control_load(request: Request) -> JSONResponse:
    try:
        await _require_empty_body(request)
        body = await lifecycle.load()
    except ControlError as exc:
        return _control_error(exc)
    return JSONResponse(body)


@app.post("/control/unload")
async def control_unload(request: Request) -> JSONResponse:
    try:
        await _require_empty_body(request)
        body = await lifecycle.unload()
    except ControlError as exc:
        return _control_error(exc)
    return JSONResponse(body)


@app.api_route("/v1", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def openai_proxy(request: Request, path: str = "") -> object:
    return await proxy_openai(request, path, lifecycle, http_client)


async def _require_empty_body(request: Request) -> None:
    raw = await request.body()
    if not raw.strip():
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ControlError(400, "BAD_REQUEST", "request body must be empty or {}") from exc
    if data != {}:
        raise ControlError(400, "BAD_REQUEST", "request body must be empty or {}")


def _control_error(exc: ControlError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
