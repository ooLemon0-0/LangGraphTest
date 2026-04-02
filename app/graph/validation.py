"""Deterministic validation and authorization helpers for tool execution."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.graph.planner_models import PlannerOutput, ValidatedPlan
from app.mcp_server.tool_schemas import TOOL_INPUT_MODELS


def validate_and_authorize_plan(
    planner_output: PlannerOutput,
    candidate_tools: list[dict[str, Any]],
    *,
    user_role: str = "user",
) -> ValidatedPlan:
    """Validate tool calls against manifest metadata and input schemas."""
    notes: list[str] = []
    preview: list[dict[str, Any]] = []
    candidate_index = {tool["name"]: tool for tool in candidate_tools}
    validated_calls: list[dict[str, Any]] = []
    overall_risk = "low"

    if planner_output.clarification_needed:
        return ValidatedPlan(
            intent=planner_output.intent,
            selected_tools=planner_output.selected_tools,
            confidence=planner_output.confidence,
            risk_level=planner_output.risk_level,
            clarification_needed=True,
            clarification_question=planner_output.clarification_question,
            direct_answer=planner_output.direct_answer,
            done=planner_output.done,
            notes=["Planner requested clarification before tool execution."],
        )

    for planned in planner_output.tool_calls:
        tool = candidate_index.get(planned.tool_name)
        if tool is None:
            notes.append(f"Rejected unknown or non-candidate tool: {planned.tool_name}")
            continue

        if tool["mode"] == "write" and user_role not in {"user", "admin"}:
            notes.append(f"Role {user_role} is not allowed to execute write tool {planned.tool_name}.")
            continue

        input_model = TOOL_INPUT_MODELS.get(planned.tool_name)
        normalized_arguments = planned.arguments
        if input_model is not None:
            try:
                normalized_arguments = input_model(**planned.arguments).model_dump()
            except ValidationError as exc:
                notes.append(f"Rejected invalid arguments for {planned.tool_name}: {exc}")
                continue

        validated_calls.append(
            {
                "tool_name": planned.tool_name,
                "arguments": normalized_arguments,
                "mode": tool["mode"],
                "risk_level": tool["risk_level"],
                "business_domain": tool["business_domain"],
                "permission": tool["permission"],
            }
        )
        preview.append(
            {
                "tool_name": planned.tool_name,
                "mode": tool["mode"],
                "arguments": normalized_arguments,
                "estimated_effect": estimate_effect(planned.tool_name, normalized_arguments),
            }
        )
        overall_risk = max_risk(overall_risk, str(tool["risk_level"]))

    selected_tools = [call["tool_name"] for call in validated_calls]
    need_confirmation = overall_risk == "high" and any(call["mode"] == "write" for call in validated_calls)
    if need_confirmation:
        notes.append("High-risk write operation requires explicit confirmation.")

    if not validated_calls and not planner_output.direct_answer and not planner_output.clarification_needed:
        notes.append("No validated tool calls remained after deterministic checks.")

    direct_answer = planner_output.direct_answer
    if not validated_calls and not direct_answer and not planner_output.clarification_needed:
        direct_answer = "I could not safely construct a valid tool action from that request."

    return ValidatedPlan(
        intent=planner_output.intent,
        tool_calls=validated_calls,
        selected_tools=selected_tools,
        confidence=planner_output.confidence,
        risk_level=overall_risk if validated_calls else planner_output.risk_level,
        need_confirmation=need_confirmation or planner_output.need_confirmation,
        clarification_needed=planner_output.clarification_needed,
        clarification_question=planner_output.clarification_question,
        direct_answer=direct_answer,
        done=planner_output.done,
        notes=notes,
        preview=preview,
        dry_run_supported=False,
    )


def estimate_effect(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a human-readable preview string for confirmation UIs."""
    if tool_name == "update_house_name":
        return f"Rename house {arguments.get('house_id')} to {arguments.get('new_name')}."
    if tool_name == "update_house_price":
        return f"Update house {arguments.get('house_id')} price to {arguments.get('new_price')} {arguments.get('currency', 'USD')}."
    if tool_name == "get_house_detail":
        return f"Read the details for house {arguments.get('house_id')}."
    if tool_name == "get_agent_id_by_name":
        return f"Resolve the id for agent {arguments.get('agent_name')}."
    return f"Execute {tool_name}."


def max_risk(left: str, right: str) -> str:
    """Return the higher of two coarse risk levels."""
    order = {"low": 0, "medium": 1, "high": 2}
    return left if order.get(left, 0) >= order.get(right, 0) else right
