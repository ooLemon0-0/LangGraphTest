"""Graph builder for the local orchestration flow."""

from __future__ import annotations

from functools import partial
import logging
import yaml

from langgraph.graph import END, START, StateGraph

from app.common.settings import AppSettings
from app.common.settings import resolve_from_root
from app.graph.embedding_retriever import LocalToolEmbeddingRetriever
from app.graph.nodes import (
    GraphDependencies,
    approval_step,
    execute_tools,
    fetch_tools,
    normalize_input,
    plan_action,
    render_response,
    retrieve_candidate_tools_node,
    validate_and_authorize,
)
from app.graph.router import route_after_approval, route_after_validation
from app.graph.router import route_after_execution
from app.graph.state import GraphState
from app.mcp_server.tool_schemas import ToolManifest

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover - version compatibility shim
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver


logger = logging.getLogger(__name__)


def build_graph(settings: AppSettings):
    """Construct and compile the LangGraph workflow."""
    retriever = LocalToolEmbeddingRetriever(settings.retrieval)
    if settings.retrieval.preload_on_startup:
        manifest_path = resolve_from_root(settings.mcp.manifest_path)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = ToolManifest(**(yaml.safe_load(handle) or {}))
        retriever.preload([entry.model_dump() for entry in manifest.tools])
        logger.info("Embedding retriever preload complete from %s", manifest_path)

    deps = GraphDependencies(settings, retriever=retriever)
    workflow = StateGraph(GraphState)
    workflow.add_node("normalize_input", partial(normalize_input, deps=deps))
    workflow.add_node("fetch_tools", partial(fetch_tools, deps=deps))
    workflow.add_node("retrieve_candidate_tools", partial(retrieve_candidate_tools_node, deps=deps))
    workflow.add_node("plan_action", partial(plan_action, deps=deps))
    workflow.add_node("validate_and_authorize", partial(validate_and_authorize, deps=deps))
    workflow.add_node("approval_step", partial(approval_step, deps=deps))
    workflow.add_node("execute_tools", partial(execute_tools, deps=deps))
    workflow.add_node("render_response", partial(render_response, deps=deps))

    workflow.add_edge(START, "normalize_input")
    workflow.add_edge("normalize_input", "fetch_tools")
    workflow.add_edge("fetch_tools", "retrieve_candidate_tools")
    workflow.add_edge("retrieve_candidate_tools", "plan_action")
    workflow.add_edge("plan_action", "validate_and_authorize")
    workflow.add_conditional_edges(
        "validate_and_authorize",
        route_after_validation,
        {
            "approval_step": "approval_step",
            "execute_tools": "execute_tools",
            "render_response": "render_response",
        },
    )
    workflow.add_conditional_edges(
        "approval_step",
        route_after_approval,
        {
            "execute_tools": "execute_tools",
            "render_response": "render_response",
        },
    )
    workflow.add_conditional_edges(
        "execute_tools",
        route_after_execution,
        {
            "fetch_tools": "fetch_tools",
            "render_response": "render_response",
        },
    )
    workflow.add_edge("render_response", END)
    return workflow.compile(checkpointer=MemorySaver())
