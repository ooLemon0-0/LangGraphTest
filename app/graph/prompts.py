"""Prompt builders for the local orchestration graph."""

from __future__ import annotations

from textwrap import dedent


def intent_classification_prompt(user_input: str) -> str:
    """Prompt for intent classification."""
    return dedent(
        f"""
        Classify the user request into one of these labels:
        - tool_lookup: the user is asking what tools exist or how a tool works
        - read: the user is asking to fetch or inspect data
        - write: the user is asking to change data
        - general: the user is asking for explanation or chat with no tool needed

        Return strict JSON with keys:
        - intent
        - rationale

        User request:
        {user_input}
        """
    ).strip()


def tool_planning_prompt(user_input: str, intent: str, tools_text: str) -> str:
    """Prompt for tool planning."""
    return dedent(
        f"""
        You are planning internal tool calls for a small local agent.

        Available tools:
        {tools_text}

        User request:
        {user_input}

        Intent:
        {intent}

        Return strict JSON with keys:
        - needs_tools: boolean
        - plan_notes: short string
        - tool_calls: array of objects with keys tool_name, arguments, reason

        Rules:
        - Only choose write tools if the user explicitly asked for a change.
        - If no tool is needed, return an empty tool_calls array.
        - Keep arguments simple and explicit.
        """
    ).strip()


def final_answer_prompt(user_input: str, tool_results_text: str, review_notes: list[str]) -> str:
    """Prompt for answer synthesis."""
    joined_notes = "\n".join(f"- {note}" for note in review_notes) or "- No review notes."
    return dedent(
        f"""
        Write a concise final answer for the user.

        User request:
        {user_input}

        Tool results:
        {tool_results_text}

        Review notes:
        {joined_notes}

        Rules:
        - Clearly say when data is mock data.
        - Be direct and short.
        """
    ).strip()
