"""Structured planner schemas and related trace models.

These models are kept separate so the planner contract can evolve
independently for retrieval, SFT data collection, and RL signals.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallPlan(BaseModel):
    """One tool call proposed by the planner."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PlannerOutput(BaseModel):
    """Strict planner schema returned by the primary LLM call."""

    intent: Literal["tool_lookup", "read", "write", "general"] = "general"
    selected_tools: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    confidence: float = 0.0
    risk_level: Literal["low", "medium", "high"] = "low"
    need_confirmation: bool = False
    clarification_needed: bool = False
    clarification_question: str = ""
    direct_answer: str = ""
    done: bool = False


class PlannerTrace(BaseModel):
    """Trace payload for one planner invocation."""

    purpose: str
    max_tokens: int
    temperature: float
    latency_seconds: float
    parsed_ok: bool
    repaired: bool = False
    raw_output: str = ""
    repair_raw_output: str = ""


class ApprovalPayload(BaseModel):
    """Interrupt payload exposed to the caller for write confirmation."""

    status: Literal["pending_confirmation"] = "pending_confirmation"
    question: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "high"
    preview: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidatedPlan(BaseModel):
    """Deterministic, authorized execution plan."""

    intent: Literal["tool_lookup", "read", "write", "general"] = "general"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    risk_level: Literal["low", "medium", "high"] = "low"
    need_confirmation: bool = False
    clarification_needed: bool = False
    clarification_question: str = ""
    direct_answer: str = ""
    done: bool = False
    notes: list[str] = Field(default_factory=list)
    preview: list[dict[str, Any]] = Field(default_factory=list)
    dry_run_supported: bool = False
