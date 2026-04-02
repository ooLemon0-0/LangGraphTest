"""Routing helpers for the graph."""

from __future__ import annotations

from app.graph.state import GraphState


def route_after_validation(state: GraphState) -> str:
    """Choose the next edge after deterministic validation.

    Main-path semantics:
    - clarification or done-without-tool => render_response
    - approval policy hit => approval_step
    - otherwise, validated low-risk / safe tool call => execute_tools
    """
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


def route_after_execution(state: GraphState) -> str:
    """Choose whether to continue looping after a tool result."""
    if state.get("iteration_count", 0) >= state.get("max_iterations", 4):
        return "render_response"
    return "fetch_tools"
