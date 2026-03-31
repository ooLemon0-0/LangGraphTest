"""State definitions for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.common.schemas import ChatMessage


class PlannedToolCall(TypedDict):
    """One planned tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    reason: str


class GraphState(TypedDict, total=False):
    """Shared graph state."""

    trace_id: str
    messages: list[ChatMessage]
    normalized_user_input: str
    intent: Literal["tool_lookup", "read", "write", "general"]
    plan_notes: str
    tool_plan: list[PlannedToolCall]
    tool_results: list[dict[str, Any]]
    review_notes: list[str]
    final_answer: str
