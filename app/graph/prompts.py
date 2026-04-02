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
        你是一个本地 tool-use 系统的规划层。

        用户请求：
        {user_input}

        候选工具：
        {candidate_tools_text}

        请严格返回一个 JSON 对象，并且只能包含这些键：
        - intent: 取值只能是 ["tool_lookup", "read", "write", "general"]
        - selected_tools: 工具名数组
        - tool_calls: 数组，元素包含 tool_name 和 arguments
        - confidence: 0 到 1 之间的小数
        - risk_level: 取值只能是 ["low", "medium", "high"]
        - need_confirmation: 布尔值
        - clarification_needed: 布尔值
        - clarification_question: 字符串
        - direct_answer: 字符串

        规则：
        - 除非确实需要多个工具，否则优先输出 0 个或 1 个工具调用。
        - 只有当用户明确要求“修改 / 更新 / 写入”时，才能选择 write 工具。
        - 如果用户意图不清楚，请设置 clarification_needed=true，并给出一句简短澄清问题。
        - 如果不需要工具，请保持 tool_calls 为空，并在 direct_answer 中直接回答。
        - 只能输出合法 JSON，不能输出 markdown，不能输出解释，不能输出代码块。
        """
    ).strip()
