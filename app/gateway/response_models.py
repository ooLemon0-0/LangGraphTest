"""Response models for gateway endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolManifestListResponse(BaseModel):
    """Response model for `/v1/tools`."""

    tools: list[dict[str, Any]] = Field(default_factory=list)


class TraceResponse(BaseModel):
    """Stored graph trace snapshot."""

    trace_id: str
    state: dict[str, Any]
