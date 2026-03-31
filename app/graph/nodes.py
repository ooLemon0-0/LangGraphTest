"""LangGraph node implementations."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from app.common.schemas import ChatMessage, ToolInvocationRequest
from app.common.settings import AppSettings
from app.graph.prompts import (
    final_answer_prompt,
    intent_classification_prompt,
    tool_planning_prompt,
)
from app.graph.state import GraphState, PlannedToolCall
from app.llm_client.openai_compatible import OpenAICompatibleClient


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


async def normalize_input(state: GraphState, _: GraphDependencies) -> GraphState:
    """Normalize the last user message into a plain string."""
    user_messages = [message for message in state["messages"] if message.role == "user"]
    latest = user_messages[-1].content if user_messages else ""
    normalized = " ".join(latest.strip().split())
    return {"normalized_user_input": normalized}


async def classify_intent(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Classify the user request."""
    user_input = state.get("normalized_user_input", "")
    if not user_input:
        return {"intent": "general"}

    prompt = intent_classification_prompt(user_input)
    messages = [ChatMessage(role="system", content=prompt)]
    try:
        response = await deps.llm_client.chat_json(messages)
        intent = response.get("intent", "general")
        if intent not in {"tool_lookup", "read", "write", "general"}:
            intent = "general"
        return {"intent": intent, "plan_notes": response.get("rationale", "")}
    except Exception as exc:
        logger.warning("Intent classification fallback used: %s", exc)
        lowered = user_input.lower()
        if any(keyword in lowered for keyword in ["rename", "update", "change", "set price"]):
            return {"intent": "write", "plan_notes": "Fallback keyword classification."}
        if any(keyword in lowered for keyword in ["house", "agent", "tool"]):
            return {"intent": "read", "plan_notes": "Fallback keyword classification."}
        return {"intent": "general", "plan_notes": "Fallback default classification."}


async def plan_tool_calls(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Plan internal tool calls based on user input and tool manifest."""
    user_input = state.get("normalized_user_input", "")
    intent = state.get("intent", "general")

    async with httpx.AsyncClient(timeout=deps.settings.mcp.request_timeout_seconds) as client:
        response = await client.get(f"{deps.mcp_base_url}/tools")
        response.raise_for_status()
        tools = response.json()["tools"]

    tools_text = json.dumps(
        [
            {
                "name": item["name"],
                "description": item["description"],
                "input_fields": item["input_fields"],
                "mode": item["mode"],
                "risk_level": item["risk_level"],
            }
            for item in tools
        ],
        indent=2,
    )

    prompt = tool_planning_prompt(user_input=user_input, intent=intent, tools_text=tools_text)
    messages = [ChatMessage(role="system", content=prompt)]

    try:
        response_data = await deps.llm_client.chat_json(messages)
        planned_calls = response_data.get("tool_calls", [])
        plan: list[PlannedToolCall] = []
        for item in planned_calls:
            if not item.get("tool_name"):
                continue
            plan.append(
                {
                    "tool_name": item["tool_name"],
                    "arguments": item.get("arguments", {}),
                    "reason": item.get("reason", "Planned by LLM."),
                }
            )
        return {
            "tool_plan": plan,
            "plan_notes": response_data.get("plan_notes", state.get("plan_notes", "")),
        }
    except Exception as exc:
        logger.warning("Tool planning fallback used: %s", exc)
        return {
            "tool_plan": heuristic_tool_plan(user_input=user_input, intent=intent),
            "plan_notes": "Fallback heuristic planning.",
        }


def heuristic_tool_plan(user_input: str, intent: str) -> list[PlannedToolCall]:
    """Simple planner used if the local model response is unavailable."""
    lowered = user_input.lower()
    if "list tools" in lowered or "what tools" in lowered:
        return [{"tool_name": "list_tools", "arguments": {}, "reason": "User asked about tools."}]
    if "tool detail" in lowered and "get_house_detail" in lowered:
        return [
            {
                "tool_name": "get_tool_detail",
                "arguments": {"tool_name": "get_house_detail"},
                "reason": "User asked for tool detail.",
            }
        ]
    if intent == "read" and "house" in lowered:
        return [
            {
                "tool_name": "get_house_detail",
                "arguments": {"house_id": "house_demo_001"},
                "reason": "Fallback house read.",
            }
        ]
    return []


async def execute_tools(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Execute each planned tool against the MCP server."""
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=deps.settings.mcp.request_timeout_seconds) as client:
        for call in state.get("tool_plan", []):
            request = ToolInvocationRequest(
                tool_name=call["tool_name"],
                arguments=call["arguments"],
                trace_id=state.get("trace_id"),
            )
            response = await client.post(
                f"{deps.mcp_base_url}/invoke",
                json=request.model_dump(),
            )
            if response.is_error:
                results.append(
                    {
                        "tool_name": call["tool_name"],
                        "ok": False,
                        "error": response.text,
                    }
                )
                continue
            results.append(response.json())
    return {"tool_results": results}


async def review_results(state: GraphState, _: GraphDependencies) -> GraphState:
    """Review tool outputs before final answer synthesis."""
    notes: list[str] = []
    tool_results = state.get("tool_results", [])
    if not tool_results:
        notes.append("No tool calls were needed for this response.")
    for result in tool_results:
        if result.get("mock"):
            notes.append(f"{result['tool_name']} returned mock data.")
        if not result.get("ok", False):
            notes.append(f"{result['tool_name']} failed and needs attention.")
    return {"review_notes": notes}


async def finalize(state: GraphState, deps: GraphDependencies) -> GraphState:
    """Create the final user-facing answer."""
    user_input = state.get("normalized_user_input", "")
    tool_results_text = json.dumps(state.get("tool_results", []), indent=2)
    review_notes = state.get("review_notes", [])
    prompt = final_answer_prompt(user_input, tool_results_text, review_notes)

    try:
        answer = await deps.llm_client.chat([ChatMessage(role="system", content=prompt)])
    except Exception as exc:
        logger.warning("Finalize fallback used: %s", exc)
        if state.get("tool_results"):
            answer = f"Completed with mock tool results: {tool_results_text}"
        else:
            answer = "No tools were needed. This local scaffold is ready for custom business logic."
    return {"final_answer": answer}
