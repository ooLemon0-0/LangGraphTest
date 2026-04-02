"""Prompt builders for the local orchestration graph."""

from __future__ import annotations

from textwrap import dedent


def plan_action_prompt(user_input: str, candidate_tools_text: str) -> str:
    """Prompt for the single structured planning call.

    This prompt is intentionally compact because planner latency matters.
    The output contract is aligned with `PlannerOutput` for future SFT/RL data.
    """
    return dedent(
        f"""
        You are the planning layer for a local tool-use system.

        User request:
        {user_input}

        Candidate tools:
        {candidate_tools_text}

        Return exactly one JSON object with these keys:
        - intent: one of ["tool_lookup", "read", "write", "general"]
        - selected_tools: array of tool names
        - tool_calls: array of objects with keys tool_name and arguments
        - confidence: number between 0 and 1
        - risk_level: one of ["low", "medium", "high"]
        - need_confirmation: boolean
        - clarification_needed: boolean
        - clarification_question: string
        - direct_answer: string

        Rules:
        - Prefer zero or one tool call unless multiple tools are clearly required.
        - Only choose write tools if the user explicitly asked to change data.
        - If the request is ambiguous, set clarification_needed=true and ask one concise question.
        - If no tool is needed, keep tool_calls empty and fill direct_answer.
        - Output valid JSON only. No markdown, no prose, no code fences.
        """
    ).strip()
