"""Meta tools for discovering the internal tool layer."""

from __future__ import annotations

from typing import Any

from app.mcp_server.tool_schemas import ToolExecutionResult, ToolManifest


def list_tools(manifest: ToolManifest, _: dict[str, Any]) -> ToolExecutionResult:
    """Return all tools from the manifest."""
    tools = [
        {
            "name": entry.name,
            "description": entry.description,
            "mode": entry.mode,
            "risk_level": entry.risk_level,
        }
        for entry in manifest.tools
    ]
    return ToolExecutionResult(
        tool_name="list_tools",
        ok=True,
        result={"tools": tools},
    )


def get_tool_detail(manifest: ToolManifest, payload: dict[str, Any]) -> ToolExecutionResult:
    """Return the full manifest entry for a named tool."""
    tool_name = payload.get("tool_name", "")
    entry = next((item for item in manifest.tools if item.name == tool_name), None)
    if entry is None:
        return ToolExecutionResult(
            tool_name="get_tool_detail",
            ok=False,
            error=f"Tool not found: {tool_name}",
        )
    return ToolExecutionResult(
        tool_name="get_tool_detail",
        ok=True,
        result={"tool": entry.model_dump()},
    )
