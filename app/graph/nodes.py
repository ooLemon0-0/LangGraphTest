"""LangGraph node implementations."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

import httpx
from pydantic import ValidationError

from app.common.schemas import ChatMessage, ToolInvocationRequest
from app.common.settings import AppSettings
from app.graph.planner_models import ApprovalPayload, PlannerOutput
from app.graph.prompts import plan_action_prompt
from app.graph.state import GraphState
from app.graph.tool_retrieval import retrieve_candidate_tools
from app.graph.validation import validate_and_authorize_plan
from app.llm_client.openai_compatible import OpenAICompatibleClient

try:
    from langgraph.types import interrupt
except ImportError:  # pragma: no cover - graph runtime compatibility
    interrupt = None


logger = logging.getLogger(__name__)


class GraphDependencies:
    """Dependencies shared across graph nodes."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.llm_client = OpenAICompatibleClient(settings.llm)

    @property
    def mcp_base_url(self) -> str:
        """Return the MCP service base URL."""
        service = self.settings.mcp.service
        return f"http://{service.host}:{service.port}"


async def normalize_input(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Normalize the last user message into a plain string."""
    _ = deps
    user_messages = [message for message in state["messages"] if message.role == "user"]
    latest = user_messages[-1].content if user_messages else ""
    normalized = " ".join(latest.strip().split())
    return {
        "normalized_user_input": normalized,
        "response_notes": [],
        "execution_trace": [],
    }


async def fetch_tools(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Fetch the full tool manifest from MCP on every request."""
    started = perf_counter()
    async with httpx.AsyncClient(timeout=deps.settings.mcp.request_timeout_seconds) as client:
        response = await client.get(f"{deps.mcp_base_url}/tools")
        response.raise_for_status()
        tools = response.json()["tools"]
    latency_seconds = perf_counter() - started
    logger.info("Fetched %s tools from MCP in %.3fs", len(tools), latency_seconds)
    return {
        "all_tools_snapshot": tools,
        "execution_trace": [
            {
                "step": "fetch_tools",
                "all_tools_count": len(tools),
                "latency_seconds": round(latency_seconds, 3),
            }
        ],
    }


async def retrieve_candidate_tools_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Reduce the full tool list down to a small candidate set."""
    _ = deps
    user_input = state.get("normalized_user_input", "")
    all_tools = state.get("all_tools_snapshot", [])
    candidates = retrieve_candidate_tools(user_input=user_input, all_tools=all_tools, top_k=5)
    candidate_names = [tool["name"] for tool in candidates]
    logger.info(
        "Tool retrieval selected %s/%s candidates: %s",
        len(candidates),
        len(all_tools),
        candidate_names,
    )
    execution_trace = list(state.get("execution_trace", []))
    execution_trace.append(
        {
            "step": "retrieve_candidate_tools",
            "all_tools_count": len(all_tools),
            "candidate_tools_count": len(candidates),
            "candidate_names": candidate_names,
        }
    )
    return {
        "candidate_tools": candidates,
        "execution_trace": execution_trace,
    }


async def plan_action(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Run the single primary LLM planning call with strict schema validation."""
    user_input = state.get("normalized_user_input", "")
    candidate_tools = state.get("candidate_tools", [])
    planner_payload = {
        "user_input": user_input,
        "candidate_tools": candidate_tools,
    }
    prompt = plan_action_prompt(user_input=user_input, candidate_tools_text=json.dumps(candidate_tools, indent=2))
    started = perf_counter()
    parsed, trace = await deps.llm_client.chat_structured(
        [ChatMessage(role="system", content=prompt)],
        schema=PlannerOutput,
        purpose="plan_action",
        max_tokens=deps.settings.llm.planner_max_tokens,
        temperature=deps.settings.llm.planner_temperature,
    )
    latency_seconds = perf_counter() - started
    logger.info(
        "LLM plan_action completed in %.3fs (parsed=%s repaired=%s max_tokens=%s)",
        latency_seconds,
        trace.parsed_ok,
        trace.repaired,
        trace.max_tokens,
    )
    execution_trace = list(state.get("execution_trace", []))
    execution_trace.append(
        {
            "step": "plan_action",
            "latency_seconds": round(latency_seconds, 3),
            "parsed_ok": trace.parsed_ok,
            "repaired": trace.repaired,
            "selected_tools": parsed.selected_tools,
        }
    )
    return {
        "intent": parsed.intent,
        "planner_prompt_payload": planner_payload,
        "planner_output": parsed.model_dump(),
        "planner_raw_output": trace.raw_output,
        "planner_trace": trace.model_dump(),
        "execution_trace": execution_trace,
    }


async def validate_and_authorize(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Validate planned tool calls and enforce deterministic safety checks."""
    _ = deps
    planner_trace = state.get("planner_trace", {})
    raw_output = state.get("planner_raw_output", "")
    planner_output = PlannerOutput.model_validate(state.get("planner_output", {}))

    validated = validate_and_authorize_plan(
        planner_output=planner_output,
        candidate_tools=state.get("candidate_tools", []),
        user_role=str(state.get("metadata", {}).get("user_role", "user")),
    )
    execution_trace = list(state.get("execution_trace", []))
    execution_trace.append(
        {
            "step": "validate_and_authorize",
            "validated_tool_calls": len(validated.tool_calls),
            "risk_level": validated.risk_level,
            "need_confirmation": validated.need_confirmation,
            "clarification_needed": validated.clarification_needed,
        }
    )
    return {
        "validated_plan": validated.model_dump(),
        "tool_plan": validated.tool_calls,
        "intent": validated.intent,
        "need_confirmation": validated.need_confirmation,
        "clarification_needed": validated.clarification_needed,
        "clarification_question": validated.clarification_question,
        "response_notes": validated.notes,
        "execution_trace": execution_trace,
    }


async def approval_step(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Pause for explicit approval before executing a high-risk write."""
    _ = deps
    validated_plan = state.get("validated_plan", {})
    payload = ApprovalPayload(
        question="Please confirm this high-risk write action before execution.",
        tool_calls=validated_plan.get("tool_calls", []),
        risk_level=validated_plan.get("risk_level", "high"),
        preview=validated_plan.get("preview", []),
        notes=validated_plan.get("notes", []),
    )

    metadata = state.get("metadata", {})
    resume_value = metadata.get("resume")
    if resume_value is None:
        if interrupt is None:
            return {
                "approval_status": "pending",
                "approval_payload": payload.model_dump(),
                "need_confirmation": True,
                "final_answer": "Confirmation is required before executing this write action.",
            }
        resume_value = interrupt(payload.model_dump())

    approved = bool(resume_value)
    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_payload": payload.model_dump(),
        "response_notes": list(state.get("response_notes", []))
        + (["User approved write action."] if approved else ["User rejected write action."]),
    }


async def execute_tools(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Execute each validated tool against the MCP server."""
    results: list[dict[str, Any]] = []
    execution_trace = list(state.get("execution_trace", []))
    async with httpx.AsyncClient(timeout=deps.settings.mcp.request_timeout_seconds) as client:
        for call in state.get("tool_plan", []):
            request = ToolInvocationRequest(
                tool_name=call["tool_name"],
                arguments=call["arguments"],
                trace_id=state.get("trace_id"),
            )
            started = perf_counter()
            response = await client.post(
                f"{deps.mcp_base_url}/invoke",
                json=request.model_dump(),
            )
            latency_seconds = perf_counter() - started
            if response.is_error:
                results.append(
                    {
                        "tool_name": call["tool_name"],
                        "ok": False,
                        "error": response.text,
                    }
                )
            else:
                results.append(response.json())
            execution_trace.append(
                {
                    "step": "execute_tool",
                    "tool_name": call["tool_name"],
                    "latency_seconds": round(latency_seconds, 3),
                    "ok": not response.is_error,
                }
            )
    return {"tool_results": results, "execution_trace": execution_trace}


async def render_response(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Render the final response with template-first logic.

    Future summarize hooks for complex multi-tool outputs can be inserted here.
    """
    _ = deps
    validated_plan = state.get("validated_plan", {})
    tool_results = state.get("tool_results", [])
    notes = list(state.get("response_notes", []))

    if state.get("clarification_needed"):
        answer = validated_plan.get("clarification_question") or "I need one clarification before proceeding."
    elif state.get("approval_status") == "pending":
        answer = "Confirmation is required before executing this write action."
    elif state.get("approval_status") == "rejected":
        answer = "The requested write action was cancelled."
    elif validated_plan.get("direct_answer") and not tool_results and not validated_plan.get("tool_calls"):
        answer = validated_plan["direct_answer"]
    elif len(tool_results) == 1:
        answer = render_single_tool_result(tool_results[0])
    elif len(tool_results) > 1:
        answer = render_multi_tool_results(tool_results)
    else:
        answer = validated_plan.get("direct_answer") or "No tool execution was needed."

    if tool_results and any(item.get("mock", False) for item in tool_results):
        notes.append("One or more tool results are mock data.")

    final_response = {
        "status": derive_status(state),
        "tool_results_count": len(tool_results),
        "clarification_needed": state.get("clarification_needed", False),
        "need_confirmation": state.get("need_confirmation", False),
        "approval_status": state.get("approval_status", "not_needed"),
        "notes": notes,
    }
    return {
        "final_answer": answer,
        "final_response": final_response,
        "response_notes": notes,
    }


def render_single_tool_result(result: dict[str, Any]) -> str:
    """Template-first rendering for single-tool responses."""
    if not result.get("ok", False):
        return f"Tool {result.get('tool_name')} failed: {result.get('error', 'unknown error')}."

    tool_name = result.get("tool_name", "")
    payload = result.get("result", {})
    if tool_name == "get_house_detail":
        house = payload.get("house", {})
        return (
            f"House {house.get('house_id')} is {house.get('name')} "
            f"with price {house.get('price')} {house.get('currency', 'USD')}. "
            f"Status: {house.get('status')}. Source: {payload.get('source', 'unknown')}."
        )
    if tool_name == "update_house_price":
        return (
            f"Updated house {payload.get('house_id')} price to {payload.get('updated_price')} "
            f"{payload.get('currency', 'USD')} ({payload.get('write_status')})."
        )
    if tool_name == "update_house_name":
        return (
            f"Updated house {payload.get('house_id')} name to {payload.get('updated_name')} "
            f"({payload.get('write_status')})."
        )
    if tool_name == "list_tools":
        tools = payload.get("tools", [])
        names = ", ".join(item.get("name", "") for item in tools[:5])
        return f"Available tools include: {names}."
    return f"Tool {tool_name} completed successfully with result: {json.dumps(payload)}"


def render_multi_tool_results(results: list[dict[str, Any]]) -> str:
    """Template-first aggregation for multi-tool responses."""
    lines = []
    for item in results:
        status = "ok" if item.get("ok", False) else "failed"
        lines.append(f"{item.get('tool_name')}: {status}")
    return "Completed multiple tool actions. " + "; ".join(lines) + "."


def derive_status(state: GraphState) -> str:
    """Return a compact response status for traces and clients."""
    if state.get("clarification_needed"):
        return "needs_clarification"
    if state.get("approval_status") == "pending":
        return "pending_confirmation"
    if state.get("approval_status") == "rejected":
        return "cancelled"
    return "completed"
