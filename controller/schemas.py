from enum import Enum

from pydantic import BaseModel, Field


class ModelState(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    READY = "READY"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


class LoadRequest(BaseModel):
    model: str | None = Field(
        default=None,
        description="Optional profile/alias name. Initial version loads the configured profile only.",
    )


class LoadResponse(BaseModel):
    status: str
    state: ModelState
    model: str | None = None
    pid: int | None = None


class UnloadResponse(BaseModel):
    status: str
    state: ModelState


class StatusResponse(BaseModel):
    state: ModelState
    model: str | None
    pid: int | None
    backend: str = "llama.cpp"
    runtime_port: int


class HealthResponse(BaseModel):
    controller: str
    model_state: ModelState
    llama_server: str
