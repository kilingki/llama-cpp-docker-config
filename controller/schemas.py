from pydantic import BaseModel


class HealthResponse(BaseModel):
    controller: str
    llama_server: str
