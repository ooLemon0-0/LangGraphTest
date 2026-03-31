"""Graph builder for the local orchestration flow."""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from app.common.settings import AppSettings
from app.graph.nodes import (
    GraphDependencies,
    classify_intent,
    execute_tools,
    finalize,
    normalize_input,
    plan_tool_calls,
    review_results,
)
from app.graph.router import should_execute_tools
from app.graph.state import GraphState


def build_graph(settings: AppSettings):
    """Construct and compile the LangGraph workflow."""
    deps = GraphDependencies(settings)
    workflow = StateGraph(GraphState)
    workflow.add_node("normalize_input", partial(normalize_input, _=deps))
    workflow.add_node("classify_intent", partial(classify_intent, deps=deps))
    workflow.add_node("plan_tool_calls", partial(plan_tool_calls, deps=deps))
    workflow.add_node("execute_tools", partial(execute_tools, deps=deps))
    workflow.add_node("review_results", partial(review_results, _=deps))
    workflow.add_node("finalize", partial(finalize, deps=deps))

    workflow.add_edge(START, "normalize_input")
    workflow.add_edge("normalize_input", "classify_intent")
    workflow.add_edge("classify_intent", "plan_tool_calls")
    workflow.add_conditional_edges(
        "plan_tool_calls",
        should_execute_tools,
        {
            "execute_tools": "execute_tools",
            "review_results": "review_results",
        },
    )
    workflow.add_edge("execute_tools", "review_results")
    workflow.add_edge("review_results", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()
