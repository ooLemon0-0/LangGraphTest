"""Prompt builders for the local orchestration graph."""

from __future__ import annotations

from textwrap import dedent


def plan_action_prompt(
    user_input: str,
    candidate_tools_text: str,
    completed_tool_calls_text: str,
    tool_results_text: str,
    iteration_count: int,
) -> str:
    """Prompt for the single structured planning call."""
    return dedent(
        f"""
        你是一个本地 tool-use 系统的规划层。

        用户请求：
        {user_input}

        候选工具：
        {candidate_tools_text}

        已执行过的工具调用：
        {completed_tool_calls_text}

        当前已拿到的工具结果：
        {tool_results_text}

        当前是第 {iteration_count} 轮规划。

        你必须严格输出一个 JSON 对象，并且只能输出 JSON，不允许输出解释、markdown 或代码块。

        JSON 必须包含以下全部键：
        - intent: 只能是 "tool_lookup" / "read" / "write" / "general"
        - selected_tools: 字符串数组
        - tool_calls: 数组，元素格式为 {{"tool_name": "...", "arguments": {{...}}}}
        - confidence: 0 到 1 之间的小数
        - risk_level: 只能是 "low" / "medium" / "high"
        - need_confirmation: 布尔值
        - clarification_needed: 布尔值
        - clarification_question: 字符串
        - direct_answer: 字符串
        - done: 布尔值

        默认 JSON 模板如下，你必须按这个结构输出：
        {{
          "intent": "read",
          "selected_tools": [],
          "tool_calls": [],
          "confidence": 0.0,
          "risk_level": "low",
          "need_confirmation": false,
          "clarification_needed": false,
          "clarification_question": "",
          "direct_answer": "",
          "done": false
        }}

        规则：
        - 每一轮最多只允许输出 1 个工具调用。
        - 如果还需要下一步工具，请只规划“下一步”。
        - 只有当用户明确要求修改、更新、写入时，才能选择 write 工具。
        - 如果用户意图不清楚，请设置 clarification_needed=true，并给出一句简短澄清问题。
        - 如果已经拿到了足够结果，请保持 tool_calls 为空，并填写 direct_answer，同时 done=true。
        - 如果还没完成，请让 done=false。
        - selected_tools 应与本轮 tool_calls 中的工具名保持一致。

        复杂样例参考：
        - “给我找到张三中介名下所有房产的信息”
        这种请求通常需要多轮：
        1. 先查经纪人 ID
        2. 再查名下房源列表
        3. 再逐个查房源详情
        你必须根据“当前已拿到的工具结果”决定下一步，而不是一次性输出全部静态列表。
        """
    ).strip()
