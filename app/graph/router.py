"""Routing helpers for the graph."""

from __future__ import annotations

from app.graph.state import GraphState


def should_execute_tools(state: GraphState) -> str:
    """Choose the next edge after planning."""
    tool_plan = state.get("tool_plan", [])
    return "execute_tools" if tool_plan else "review_results"
