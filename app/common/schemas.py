"""Shared Pydantic schemas used across services."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A simple chat message."""

    role: str = Field(description="Message role such as system, user, or assistant.")
    content: str = Field(description="Natural language message content.")


class ChatRequest(BaseModel):
    """Gateway chat request."""

    messages: list[ChatMessage]
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Gateway chat response."""

    trace_id: str
    answer: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
    planner_iterations: list[dict[str, Any]] = Field(default_factory=list)
    raw_state: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Simple health payload."""

    status: str
    service: str
    version: str = "0.1.0"


class ToolInvocationRequest(BaseModel):
    """Generic tool invocation request."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class ToolInvocationResponse(BaseModel):
    """Generic tool invocation response."""

    tool_name: str
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    mock: bool = True
