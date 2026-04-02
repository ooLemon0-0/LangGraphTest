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
except ImportError:  # pragma: no cover
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
        "completed_tool_calls": [],
        "iteration_count": 0,
        "max_iterations": 4,
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
        "execution_trace": [{"step": "fetch_tools", "all_tools_count": len(tools), "latency_seconds": round(latency_seconds, 3)}],
    }


async def retrieve_candidate_tools_node(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Reduce the full tool list down to a small candidate set."""
    user_input = state.get("normalized_user_input", "")
    all_tools = state.get("all_tools_snapshot", [])

    embedding_candidates: list[dict[str, Any]] = []
    if deps.retriever is not None and deps.settings.retrieval.enabled:
        embedding_candidates = deps.retriever.retrieve(user_input, all_tools)

    lexical_candidates = retrieve_candidate_tools(
        user_input=user_input,
        all_tools=all_tools,
        top_k=deps.settings.retrieval.top_k,
    )
    candidates = merge_candidate_lists(
        primary=embedding_candidates,
        secondary=lexical_candidates,
        top_k=deps.settings.retrieval.top_k,
    ) if embedding_candidates else lexical_candidates

    candidate_names = [tool["name"] for tool in candidates]
    logger.info("Tool retrieval selected %s/%s candidates: %s", len(candidates), len(all_tools), candidate_names)

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
    return {"candidate_tools": candidates, "execution_trace": execution_trace}


async def plan_action(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Run the single primary LLM planning call with strict schema validation.

    SFT hook:
    - planner_prompt_payload
    - planner_raw_output
    - planner_trace
    - planner_output
    are all kept in graph state on purpose so future SFT / preference training
    can collect aligned planner examples without changing the external API.
    """
    user_input = state.get("normalized_user_input", "")
    candidate_tools = state.get("candidate_tools", [])
    completed_tool_calls = state.get("completed_tool_calls", [])
    tool_results = state.get("tool_results", [])
    iteration_count = state.get("iteration_count", 0) + 1
    planner_payload = {
        "user_input": user_input,
        "candidate_tools": candidate_tools,
        "completed_tool_calls": completed_tool_calls,
        "tool_results": tool_results,
        "iteration_count": iteration_count,
    }
    prompt = plan_action_prompt(
        user_input=user_input,
        candidate_tools_text=json.dumps(candidate_tools, ensure_ascii=False, indent=2),
        completed_tool_calls_text=json.dumps(completed_tool_calls, ensure_ascii=False, indent=2),
        tool_results_text=json.dumps(tool_results, ensure_ascii=False, indent=2),
        iteration_count=iteration_count,
    )

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
        parsed = heuristic_planner_output_multi_step(
            user_input=user_input,
            candidate_tools=candidate_tools,
            completed_tool_calls=completed_tool_calls,
            tool_results=tool_results,
        )
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
            "iteration_count": iteration_count,
        }
    )
    return {
        "intent": parsed.intent,
        "planner_prompt_payload": planner_payload,
        "planner_output": parsed.model_dump(),
        "planner_raw_output": trace.raw_output,
        "planner_trace": trace.model_dump(),
        "execution_trace": execution_trace,
        "iteration_count": iteration_count,
    }


def heuristic_planner_output(user_input: str, candidate_tools: list[dict[str, Any]]) -> PlannerOutput:
    """Deterministic fallback planner when structured LLM output is unavailable.

    SFT hook:
    This fallback remains intentionally simple and inspectable so it can serve
    as a bootstrap policy while we collect high-quality planner traces.
    """
    lowered = user_input.lower()
    candidate_names = {tool["name"] for tool in candidate_tools}

    tool_lookup_terms = ["有哪些工具", "什么工具", "工具", "tool"]
    write_terms = ["价格", "改价", "修改", "更新", "重命名", "改名"]
    read_terms = ["房源", "详情", "详细信息", "查看", "查询", "house"]

    if any(term in lowered for term in tool_lookup_terms):
        tool_name = "list_tools" if "list_tools" in candidate_names else ""
        tool_calls = [{"tool_name": tool_name, "arguments": {}}] if tool_name else []
        return PlannerOutput(
            intent="tool_lookup",
            selected_tools=[tool_name] if tool_name else [],
            tool_calls=tool_calls,
            confidence=0.35,
            risk_level="low",
            direct_answer="" if tool_calls else "当前没有可用的工具清单工具。",
        )

    if any(term in lowered for term in write_terms):
        if "update_house_price" in candidate_names and "价格" in lowered:
            return PlannerOutput(
                intent="write",
                selected_tools=["update_house_price"],
                tool_calls=[{"tool_name": "update_house_price", "arguments": {"house_id": "house_demo_001", "new_price": 888888, "currency": "CNY"}}],
                confidence=0.25,
                risk_level="high",
                need_confirmation=True,
            )
        if "update_house_name" in candidate_names:
            return PlannerOutput(
                intent="write",
                selected_tools=["update_house_name"],
                tool_calls=[{"tool_name": "update_house_name", "arguments": {"house_id": "house_demo_001", "new_name": "示例房源"}}],
                confidence=0.2,
                risk_level="high",
                need_confirmation=True,
            )

    if any(term in lowered for term in read_terms) and "get_house_detail" in candidate_names:
        return PlannerOutput(
            intent="read",
            selected_tools=["get_house_detail"],
            tool_calls=[{"tool_name": "get_house_detail", "arguments": {"house_id": "house_demo_001"}}],
            confidence=0.4,
            risk_level="low",
        )

    return PlannerOutput(
        intent="general",
        selected_tools=[],
        tool_calls=[],
        confidence=0.1,
        risk_level="low",
        direct_answer="我暂时无法稳定解析这条请求，但当前系统已经完成工具检索与安全校验链路。",
        done=True,
    )


def heuristic_planner_output_multi_step(
    user_input: str,
    candidate_tools: list[dict[str, Any]],
    completed_tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> PlannerOutput:
    """Deterministic multi-step fallback planner.

    This fallback is intentionally structured so future SFT can learn a
    step-by-step tool policy from prompt/trace data instead of static lists.
    """
    lowered = user_input.lower()
    candidate_names = {tool["name"] for tool in candidate_tools}
    executed_names = [item.get("tool_name", "") for item in completed_tool_calls]

    if any(term in lowered for term in ["有哪些工具", "什么工具", "工具", "tool"]):
        if "list_tools" in candidate_names and "list_tools" not in executed_names:
            return PlannerOutput(
                intent="tool_lookup",
                selected_tools=["list_tools"],
                tool_calls=[{"tool_name": "list_tools", "arguments": {}}],
                confidence=0.35,
                risk_level="low",
                done=False,
            )
        return PlannerOutput(intent="tool_lookup", confidence=0.3, risk_level="low", direct_answer="当前工具清单已经给出。", done=True)

    agent_name = extract_agent_name(user_input)
    last_agent_id = latest_agent_id(tool_results)
    house_ids = latest_house_ids(tool_results)
    fetched_house_ids = fetched_house_detail_ids(tool_results)

    if any(term in lowered for term in ["中介", "经纪人", "agent"]) and any(term in lowered for term in ["房产", "房源", "房子"]):
        if not agent_name and not last_agent_id:
            return PlannerOutput(
                intent="read",
                confidence=0.2,
                risk_level="low",
                clarification_needed=True,
                clarification_question="请告诉我你想查询哪位中介的名字。",
                done=True,
            )
        if not last_agent_id and "get_agent_id_by_name" in candidate_names:
            return PlannerOutput(
                intent="read",
                selected_tools=["get_agent_id_by_name"],
                tool_calls=[{"tool_name": "get_agent_id_by_name", "arguments": {"agent_name": agent_name}}],
                confidence=0.45,
                risk_level="low",
                done=False,
            )
        if last_agent_id and not house_ids and "get_houses_by_agent_id" in candidate_names:
            return PlannerOutput(
                intent="read",
                selected_tools=["get_houses_by_agent_id"],
                tool_calls=[{"tool_name": "get_houses_by_agent_id", "arguments": {"agent_id": last_agent_id}}],
                confidence=0.5,
                risk_level="low",
                done=False,
            )
        remaining_house_ids = [house_id for house_id in house_ids if house_id not in fetched_house_ids]
        if remaining_house_ids and "get_house_detail" in candidate_names:
            return PlannerOutput(
                intent="read",
                selected_tools=["get_house_detail"],
                tool_calls=[{"tool_name": "get_house_detail", "arguments": {"house_id": remaining_house_ids[0]}}],
                confidence=0.55,
                risk_level="low",
                done=False,
            )
        if house_ids and not remaining_house_ids:
            return PlannerOutput(
                intent="read",
                confidence=0.55,
                risk_level="low",
                direct_answer="已拿到该中介名下房源的完整信息。",
                done=True,
            )

    if any(term in lowered for term in ["价格", "改价", "修改", "更新", "重命名", "改名"]):
        if "update_house_price" in candidate_names and "价格" in lowered:
            return PlannerOutput(
                intent="write",
                selected_tools=["update_house_price"],
                tool_calls=[{"tool_name": "update_house_price", "arguments": {"house_id": "house_demo_001", "new_price": 888888, "currency": "CNY"}}],
                confidence=0.25,
                risk_level="high",
                need_confirmation=True,
                done=False,
            )
        if "update_house_name" in candidate_names:
            return PlannerOutput(
                intent="write",
                selected_tools=["update_house_name"],
                tool_calls=[{"tool_name": "update_house_name", "arguments": {"house_id": "house_demo_001", "new_name": "示例房源"}}],
                confidence=0.2,
                risk_level="high",
                need_confirmation=True,
                done=False,
            )

    if any(term in lowered for term in ["房源", "详情", "详细信息", "查看", "查询", "house"]) and "get_house_detail" in candidate_names:
        if not fetched_house_ids:
            return PlannerOutput(
                intent="read",
                selected_tools=["get_house_detail"],
                tool_calls=[{"tool_name": "get_house_detail", "arguments": {"house_id": "house_demo_001"}}],
                confidence=0.4,
                risk_level="low",
                done=False,
            )
        return PlannerOutput(intent="read", confidence=0.35, risk_level="low", direct_answer="已拿到房源详情。", done=True)

    return heuristic_planner_output(user_input=user_input, candidate_tools=candidate_tools)


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
        "tool_plan": validated.tool_calls[:1],
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
        question="请确认是否执行这个高风险写操作。",
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
                "final_answer": "执行这个写操作前需要你确认。",
            }
        resume_value = interrupt(payload.model_dump())

    approved = bool(resume_value)
    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_payload": payload.model_dump(),
        "response_notes": list(state.get("response_notes", [])) + (["用户已确认执行写操作。"] if approved else ["用户已拒绝执行写操作。"]),
    }


async def execute_tools(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Execute the next validated tool against the MCP server."""
    results = list(state.get("tool_results", []))
    completed_tool_calls = list(state.get("completed_tool_calls", []))
    execution_trace = list(state.get("execution_trace", []))
    next_call = next(iter(state.get("tool_plan", [])), None)
    if next_call is None:
        return {"tool_results": results, "execution_trace": execution_trace}

    async with httpx.AsyncClient(timeout=deps.settings.mcp.request_timeout_seconds) as client:
        request = ToolInvocationRequest(
            tool_name=next_call["tool_name"],
            arguments=next_call["arguments"],
            trace_id=state.get("trace_id"),
        )
        started = perf_counter()
        response = await client.post(f"{deps.mcp_base_url}/invoke", json=request.model_dump())
        latency_seconds = perf_counter() - started
        if response.is_error:
            results.append({"tool_name": next_call["tool_name"], "ok": False, "error": response.text})
        else:
            results.append(response.json())
        completed_tool_calls.append(next_call)
        execution_trace.append(
            {
                "step": "execute_tool",
                "tool_name": next_call["tool_name"],
                "latency_seconds": round(latency_seconds, 3),
                "ok": not response.is_error,
            }
        )
    return {"tool_results": results, "completed_tool_calls": completed_tool_calls, "execution_trace": execution_trace, "need_confirmation": False}


async def render_response(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Render the final response with template-first logic.

    We keep the default path deterministic so the common request still uses one
    LLM call. A future summarize hook can be attached here after SFT/RL data
    validates that a second model call is worth the extra latency.
    """
    _ = deps
    validated_plan = state.get("validated_plan", {})
    tool_results = state.get("tool_results", [])
    notes = list(state.get("response_notes", []))
    prefers_chinese = contains_chinese(state.get("normalized_user_input", ""))

    if state.get("clarification_needed"):
        answer = validated_plan.get("clarification_question") or ("我需要先确认一个细节再继续。" if prefers_chinese else "I need one clarification before proceeding.")
    elif state.get("approval_status") == "pending":
        answer = "执行这个写操作前需要你确认。" if prefers_chinese else "Confirmation is required before executing this write action."
    elif state.get("approval_status") == "rejected":
        answer = "已取消这次写操作。" if prefers_chinese else "The requested write action was cancelled."
    elif validated_plan.get("direct_answer") and not state.get("tool_plan"):
        answer = validated_plan["direct_answer"]
    elif len(tool_results) == 1:
        answer = render_single_tool_result(tool_results[0], prefers_chinese=prefers_chinese)
    elif len(tool_results) > 1:
        answer = render_multi_tool_results(tool_results, prefers_chinese=prefers_chinese)
    else:
        answer = validated_plan.get("direct_answer") or ("这次请求不需要执行工具。" if prefers_chinese else "No tool execution was needed.")

    if tool_results and any(item.get("mock", False) for item in tool_results):
        notes.append("存在 mock 数据结果。" if prefers_chinese else "One or more tool results are mock data.")

    final_response = {
        "status": derive_status(state),
        "tool_results_count": len(tool_results),
        "clarification_needed": state.get("clarification_needed", False),
        "need_confirmation": state.get("need_confirmation", False),
        "approval_status": state.get("approval_status", "not_needed"),
        "notes": notes,
    }
    return {"final_answer": answer, "final_response": final_response, "response_notes": notes}


def render_single_tool_result(result: dict[str, Any], *, prefers_chinese: bool) -> str:
    """Template-first rendering for single-tool responses."""
    if not result.get("ok", False):
        return (
            f"工具 {result.get('tool_name')} 执行失败：{result.get('error', '未知错误')}。"
            if prefers_chinese
            else f"Tool {result.get('tool_name')} failed: {result.get('error', 'unknown error')}."
        )

    tool_name = result.get("tool_name", "")
    payload = result.get("result", {})
    if tool_name == "get_house_detail":
        house = payload.get("house", {})
        if prefers_chinese:
            return (
                f"房源 {house.get('house_id')} 的名称是 {house.get('name')}，"
                f"价格是 {house.get('price')} {house.get('currency', 'USD')}，"
                f"当前状态是 {house.get('status')}。数据来源：{payload.get('source', 'unknown')}。"
            )
        return (
            f"House {house.get('house_id')} is {house.get('name')} "
            f"with price {house.get('price')} {house.get('currency', 'USD')}. "
            f"Status: {house.get('status')}. Source: {payload.get('source', 'unknown')}."
        )
    if tool_name == "update_house_price":
        if prefers_chinese:
            return (
                f"已将房源 {payload.get('house_id')} 的价格更新为 "
                f"{payload.get('updated_price')} {payload.get('currency', 'USD')}，"
                f"执行状态：{payload.get('write_status')}。"
            )
        return (
            f"Updated house {payload.get('house_id')} price to {payload.get('updated_price')} "
            f"{payload.get('currency', 'USD')} ({payload.get('write_status')})."
        )
    if tool_name == "update_house_name":
        if prefers_chinese:
            return (
                f"已将房源 {payload.get('house_id')} 的名称更新为 "
                f"{payload.get('updated_name')}，执行状态：{payload.get('write_status')}。"
            )
        return (
            f"Updated house {payload.get('house_id')} name to {payload.get('updated_name')} "
            f"({payload.get('write_status')})."
        )
    if tool_name == "list_tools":
        tools = payload.get("tools", [])
        names = ", ".join(item.get("name", "") for item in tools[:5])
        return f"当前可用工具包括：{names}。" if prefers_chinese else f"Available tools include: {names}."
    return (
        f"工具 {tool_name} 已执行成功，结果为：{json.dumps(payload, ensure_ascii=False)}"
        if prefers_chinese
        else f"Tool {tool_name} completed successfully with result: {json.dumps(payload)}"
    )


def render_multi_tool_results(results: list[dict[str, Any]], *, prefers_chinese: bool) -> str:
    """Template-first aggregation for multi-tool responses."""
    lines = []
    for item in results:
        status = ("成功" if item.get("ok", False) else "失败") if prefers_chinese else ("ok" if item.get("ok", False) else "failed")
        lines.append(f"{item.get('tool_name')}: {status}")
    return ("已完成多个工具操作：" + "；".join(lines) + "。") if prefers_chinese else ("Completed multiple tool actions. " + "; ".join(lines) + ".")


def derive_status(state: GraphState) -> str:
    """Return a compact response status for traces and clients."""
    if state.get("clarification_needed"):
        return "needs_clarification"
    if state.get("approval_status") == "pending":
        return "pending_confirmation"
    if state.get("approval_status") == "rejected":
        return "cancelled"
    return "completed"


def contains_chinese(text: str) -> bool:
    """Return True when the input contains Chinese characters."""
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def extract_agent_name(user_input: str) -> str:
    """Extract a likely agent name from common Chinese phrasing."""
    markers = ["中介", "经纪人", "agent"]
    for marker in markers:
        index = user_input.find(marker)
        if index > 0:
            prefix = user_input[:index].strip(" 给我找到查询请帮我帮我查一下的名下所有房产信息")
            if prefix:
                return prefix.strip()
    return ""


def latest_agent_id(tool_results: list[dict[str, Any]]) -> str:
    """Return the most recent agent id obtained from tool results."""
    for result in reversed(tool_results):
        payload = result.get("result", {})
        if "agent_id" in payload:
            return str(payload["agent_id"])
    return ""


def latest_house_ids(tool_results: list[dict[str, Any]]) -> list[str]:
    """Return house ids from the latest house list result."""
    for result in reversed(tool_results):
        payload = result.get("result", {})
        houses = payload.get("houses")
        if isinstance(houses, list):
            return [str(item.get("house_id")) for item in houses if item.get("house_id")]
    return []


def fetched_house_detail_ids(tool_results: list[dict[str, Any]]) -> list[str]:
    """Return all house ids already fetched via get_house_detail."""
    ids: list[str] = []
    for result in tool_results:
        if result.get("tool_name") != "get_house_detail":
            continue
        house = result.get("result", {}).get("house", {})
        house_id = house.get("house_id")
        if house_id:
            ids.append(str(house_id))
    return ids


def merge_candidate_lists(*, primary: list[dict[str, Any]], secondary: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
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
