"""YAML-backed application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


class ProjectSettings(BaseModel):
    """Top-level project metadata."""

    name: str
    environment_name: str
    python_version: str


class ServiceBinding(BaseModel):
    """Host and port for a service."""

    host: str
    port: int


class LLMServiceSettings(ServiceBinding):
    """Serving settings for the local LLM endpoint."""

    server_backend: str
    model_source: str
    tensor_parallel_size: int = 1
    dtype: str = "auto"


class LLMSettings(BaseModel):
    """LLM client and serving config."""

    provider: str
    model_name: str
    base_url: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_seconds: int = 60
    service: LLMServiceSettings


class MCPSettings(BaseModel):
    """MCP server config."""

    service: ServiceBinding
    manifest_path: str
    request_timeout_seconds: int = 30


class GatewaySettings(BaseModel):
    """Gateway config."""

    service: ServiceBinding
    debug: bool = True
    trace_store_limit: int = 200


class LoggingSettings(BaseModel):
    """Logging config."""

    level: str = "INFO"
    json: bool = False


class AppSettings(BaseModel):
    """The single config object shared by all services."""

    project: ProjectSettings
    llm: LLMSettings
    mcp: MCPSettings
    gateway: GatewaySettings
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: str | None = None) -> AppSettings:
    """Return cached application settings."""
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    raw = load_yaml(path)
    settings = AppSettings(**raw)
    return settings


def resolve_from_root(relative_path: str) -> Path:
    """Resolve a repository-relative path."""
    return (ROOT_DIR / relative_path).resolve()
