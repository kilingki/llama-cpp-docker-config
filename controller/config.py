import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name, default)
    return value.strip() if value is not None else default


def _truthy(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"on", "1", "true", "yes"}:
        return True
    if lowered in {"off", "0", "false", "no"}:
        return False
    if lowered == "":
        return None
    return None


@dataclass(frozen=True)
class Settings:
    controller_host: str
    controller_port: int
    llama_server_bin: str
    load_timeout_sec: float
    unload_timeout_sec: float
    llama_fit: str
    model_path: str
    model_name: str
    context_size: str
    gpu_layers: str
    cache_type_k: str
    cache_type_v: str
    flash_attn: str
    batch_size: str
    ubatch_size: str
    threads: str
    parallel: str
    llama_host: str
    llama_port: int
    mmproj_path: str
    jinja: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            controller_host=_env("CONTROLLER_HOST", "0.0.0.0"),
            controller_port=int(_env("CONTROLLER_PORT", "8000")),
            llama_server_bin=_env("LLAMA_SERVER_BIN", "/usr/local/bin/llama-server"),
            load_timeout_sec=float(_env("LOAD_TIMEOUT_SEC", "300")),
            unload_timeout_sec=float(_env("UNLOAD_TIMEOUT_SEC", "30")),
            llama_fit=_env("LLAMA_FIT", "off") or "off",
            model_path=_env("MODEL_PATH"),
            model_name=_env("MODEL_NAME"),
            context_size=_env("CONTEXT_SIZE", "32768"),
            gpu_layers=_env("GPU_LAYERS", "all"),
            cache_type_k=_env("CACHE_TYPE_K", "f16"),
            cache_type_v=_env("CACHE_TYPE_V", "f16"),
            flash_attn=_env("FLASH_ATTN", "on"),
            batch_size=_env("BATCH_SIZE", "2048"),
            ubatch_size=_env("UBATCH_SIZE", "512"),
            threads=_env("THREADS", "-1"),
            parallel=_env("PARALLEL", "1"),
            llama_host=_env("HOST", "127.0.0.1"),
            llama_port=int(_env("PORT", "8080")),
            mmproj_path=_env("MMPROJ_PATH"),
            jinja=_env("JINJA", "on"),
        )

    @property
    def llama_base_url(self) -> str:
        return f"http://{self.llama_host}:{self.llama_port}"

    def llama_server_cmd(self) -> list[str]:
        if not self.model_path:
            raise ValueError("MODEL_PATH is not set")
        if not self.llama_server_bin:
            raise ValueError("LLAMA_SERVER_BIN is not set")

        cmd = [
            self.llama_server_bin,
            "--model",
            self.model_path,
            "--ctx-size",
            self.context_size,
            "--n-gpu-layers",
            self.gpu_layers,
            "--cache-type-k",
            self.cache_type_k,
            "--cache-type-v",
            self.cache_type_v,
            "--flash-attn",
            self.flash_attn,
            "--batch-size",
            self.batch_size,
            "--ubatch-size",
            self.ubatch_size,
            "--threads",
            self.threads,
            "--parallel",
            self.parallel,
            "--host",
            self.llama_host,
            "--port",
            str(self.llama_port),
            "--fit",
            self.llama_fit,
            "--no-ui",
        ]
        if self.model_name:
            cmd.extend(["--alias", self.model_name])
        if self.mmproj_path:
            cmd.extend(["--mmproj", self.mmproj_path])

        jinja = _truthy(self.jinja)
        if jinja is True:
            cmd.append("--jinja")
        elif jinja is False:
            cmd.append("--no-jinja")
        return cmd
