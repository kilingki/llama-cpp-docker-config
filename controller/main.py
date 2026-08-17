import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request

from controller.config import Settings
from controller.lifecycle import LifecycleError, ProcessManager
from controller.proxy import proxy_openai
from controller.schemas import (
    HealthResponse,
    LoadRequest,
    LoadResponse,
    StatusResponse,
    UnloadResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("controller")

settings = Settings.from_env()
manager = ProcessManager(settings)
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
        await manager.shutdown()
        await http_client.aclose()


app = FastAPI(title="llama.cpp runtime controller", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        controller="ok",
        model_state=manager.state,
        llama_server=manager.llama_server_health_label(),
    )


@app.get("/control/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    payload = manager.status_payload()
    return StatusResponse(**payload)


@app.post("/control/load", response_model=LoadResponse)
async def load(body: LoadRequest = LoadRequest()) -> LoadResponse:
    requested = body.model
    try:
        result = await manager.load(requested)
    except LifecycleError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return LoadResponse(**result)


@app.post("/control/unload", response_model=UnloadResponse)
async def unload() -> UnloadResponse:
    try:
        result = await manager.unload()
    except LifecycleError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return UnloadResponse(**result)


@app.api_route("/v1", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def openai_proxy(request: Request, path: str = "") -> object:
    return await proxy_openai(request, path, manager, http_client)
