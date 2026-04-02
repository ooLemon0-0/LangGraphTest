"""LangGraph node implementations."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any

import httpx

from app.common.schemas import ChatMessage, ToolInvocationRequest
from app.common.settings import AppSettings
from app.graph.embedding_retriever import LocalToolEmbeddingRetriever
from app.graph.planner_models import ApprovalPayload, PlannerOutput, PlannerTrace
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

    def __init__(self, settings: AppSettings, *, retriever: LocalToolEmbeddingRetriever | None = None) -> None:
        self.settings = settings
        self.llm_client = OpenAICompatibleClient(settings.llm)
        self.retriever = retriever

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
    user_input = state.get("normalized_user_input", "")
    all_tools = state.get("all_tools_snapshot", [])
    embedding_candidates: list[dict[str, Any]] = []
    if deps.retriever is not None and deps.settings.retrieval.enabled:
        embedding_candidates = deps.retriever.retrieve(user_input, all_tools)

    candidates = retrieve_candidate_tools(
        user_input=user_input,
        all_tools=all_tools,
        top_k=deps.settings.retrieval.top_k,
    )
    if embedding_candidates:
        candidates = merge_candidate_lists(
            primary=embedding_candidates,
            secondary=candidates,
            top_k=deps.settings.retrieval.top_k,
        )
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
            "embedding_enabled": bool(deps.retriever is not None and deps.settings.retrieval.enabled),
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
    try:
        parsed, trace = await deps.llm_client.chat_structured(
            [ChatMessage(role="system", content=prompt)],
            schema=PlannerOutput,
            purpose="plan_action",
            max_tokens=deps.settings.llm.planner_max_tokens,
            temperature=deps.settings.llm.planner_temperature,
        )
    except Exception as exc:
        logger.warning("Planner structured parse fallback used: %s", exc)
        parsed = heuristic_planner_output(user_input=user_input, candidate_tools=candidate_tools)
        trace = PlannerTrace(
            purpose="plan_action",
            max_tokens=deps.settings.llm.planner_max_tokens,
            temperature=deps.settings.llm.planner_temperature,
            latency_seconds=0.0,
            parsed_ok=False,
            repaired=False,
            raw_output="",
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


def heuristic_planner_output(user_input: str, candidate_tools: list[dict[str, Any]]) -> PlannerOutput:
    """Deterministic fallback planner when structured LLM output is unavailable."""
    lowered = user_input.lower()
    candidate_names = {tool["name"] for tool in candidate_tools}

    tool_lookup_terms = [
        "\u6709\u54ea\u4e9b\u5de5\u5177",
        "\u4ec0\u4e48\u5de5\u5177",
        "\u5de5\u5177",
        "tool",
    ]
    write_terms = [
        "\u4ef7\u683c",
        "\u6539\u4ef7",
        "\u4fee\u6539",
        "\u66f4\u65b0",
        "\u91cd\u547d\u540d",
        "\u6539\u540d",
    ]
    read_terms = [
        "\u623f\u6e90",
        "\u8be6\u60c5",
        "\u8be6\u7ec6\u4fe1\u606f",
        "\u67e5\u770b",
        "\u67e5\u8be2",
        "house",
    ]

    if any(term in lowered for term in tool_lookup_terms):
        tool_name = "list_tools" if "list_tools" in candidate_names else ""
        tool_calls = [{"tool_name": tool_name, "arguments": {}}] if tool_name else []
        return PlannerOutput(
            intent="tool_lookup",
            selected_tools=[tool_name] if tool_name else [],
            tool_calls=tool_calls,
            confidence=0.35,
            risk_level="low",
            direct_answer="" if tool_calls else "\u5f53\u524d\u6ca1\u6709\u53ef\u7528\u7684\u5de5\u5177\u6e05\u5355\u5de5\u5177\u3002",
        )

    if any(term in lowered for term in write_terms):
        if "update_house_price" in candidate_names and "\u4ef7\u683c" in lowered:
            return PlannerOutput(
                intent="write",
                selected_tools=["update_house_price"],
                tool_calls=[
                    {
                        "tool_name": "update_house_price",
                        "arguments": {
                            "house_id": "house_demo_001",
                            "new_price": 888888,
                            "currency": "CNY",
                        },
                    }
                ],
                confidence=0.25,
                risk_level="high",
                need_confirmation=True,
            )
        if "update_house_name" in candidate_names:
            return PlannerOutput(
                intent="write",
                selected_tools=["update_house_name"],
                tool_calls=[
                    {
                        "tool_name": "update_house_name",
                        "arguments": {
                            "house_id": "house_demo_001",
                            "new_name": "\u793a\u4f8b\u623f\u6e90",
                        },
                    }
                ],
                confidence=0.2,
                risk_level="high",
                need_confirmation=True,
            )

    if any(term in lowered for term in read_terms):
        if "get_house_detail" in candidate_names:
            return PlannerOutput(
                intent="read",
                selected_tools=["get_house_detail"],
                tool_calls=[
                    {
                        "tool_name": "get_house_detail",
                        "arguments": {"house_id": "house_demo_001"},
                    }
                ],
                confidence=0.4,
                risk_level="low",
            )

    return PlannerOutput(
        intent="general",
        selected_tools=[],
        tool_calls=[],
        confidence=0.1,
        risk_level="low",
        direct_answer="\u6211\u6682\u65f6\u65e0\u6cd5\u7a33\u5b9a\u89e3\u6790\u8fd9\u6761\u8bf7\u6c42\uff0c\u4f46\u5f53\u524d\u7cfb\u7edf\u5df2\u7ecf\u5b8c\u6210\u5de5\u5177\u68c0\u7d22\u4e0e\u5b89\u5168\u6821\u9a8c\u94fe\u8def\u3002",
    )


async def validate_and_authorize(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Validate planned tool calls and enforce deterministic safety checks."""
    _ = deps
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
    """Render the final response with template-first logic."""
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


def merge_candidate_lists(
    *,
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Merge embedding and lexical retrieval results while preserving order."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in [primary, secondary]:
        for tool in source:
            name = tool["name"]
            if name in seen:
                continue
            merged.append(tool)
            seen.add(name)
            if len(merged) >= top_k:
                return merged
    return merged
