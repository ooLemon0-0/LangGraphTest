"""Routing helpers for the graph."""

from __future__ import annotations

from app.graph.state import GraphState


def route_after_validation(state: GraphState) -> str:
    """Choose the next edge after deterministic validation."""
    if state.get("clarification_needed"):
        return "render_response"
    if state.get("need_confirmation"):
        return "approval_step"
    if state.get("tool_plan"):
        return "execute_tools"
    return "render_response"


def route_after_approval(state: GraphState) -> str:
    """Choose the next edge after approval / interrupt resume."""
    return "execute_tools" if state.get("approval_status") == "approved" else "render_response"
