"""State definitions for the LangGraph workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.common.schemas import ChatMessage


class GraphState(TypedDict, total=False):
    """Shared graph state.

    The explicit trace fields are kept for future retrieval tuning, SFT logging,
    and reward-model style evaluation.
    """

    trace_id: str
    messages: list[ChatMessage]
    metadata: dict[str, Any]
    normalized_user_input: str

    all_tools_snapshot: list[dict[str, Any]]
    candidate_tools: list[dict[str, Any]]
    planner_prompt_payload: dict[str, Any]
    planner_output: dict[str, Any]
    planner_raw_output: str
    planner_trace: dict[str, Any]
    validated_plan: dict[str, Any]
    execution_trace: list[dict[str, Any]]

    intent: Literal["tool_lookup", "read", "write", "general"]
    tool_plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    response_notes: list[str]
    final_answer: str
    final_response: dict[str, Any]

    need_confirmation: bool
    approval_status: Literal["not_needed", "pending", "approved", "rejected"]
    approval_payload: dict[str, Any]
    clarification_needed: bool
    clarification_question: str
