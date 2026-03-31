"""Placeholder agent-related MCP tools."""

from __future__ import annotations

from app.mcp_server.tool_schemas import (
    GetAgentByHouseIdInput,
    GetAgentIdByNameInput,
    GetHousesByAgentIdInput,
    ToolExecutionResult,
)


def get_agent_id_by_name(payload: dict[str, object]) -> ToolExecutionResult:
    """Resolve a mock agent id from a name."""
    request = GetAgentIdByNameInput(**payload)
    agent_id = f"agent_{request.agent_name.strip().lower().replace(' ', '_')}"
    return ToolExecutionResult(
        tool_name="get_agent_id_by_name",
        ok=True,
        result={
            "agent_id": agent_id,
            "agent_name": request.agent_name,
            "source": "mock_data",
        },
    )


def get_houses_by_agent_id(payload: dict[str, object]) -> ToolExecutionResult:
    """Return mock houses for a given agent id."""
    request = GetHousesByAgentIdInput(**payload)
    houses = [
        {"house_id": f"{request.agent_id}_house_001", "name": "Mock Maple House"},
        {"house_id": f"{request.agent_id}_house_002", "name": "Mock River House"},
    ]
    return ToolExecutionResult(
        tool_name="get_houses_by_agent_id",
        ok=True,
        result={
            "agent_id": request.agent_id,
            "houses": houses,
            "source": "mock_data",
        },
    )


def get_agent_by_house_id(payload: dict[str, object]) -> ToolExecutionResult:
    """Return a mock agent for a house id."""
    request = GetAgentByHouseIdInput(**payload)
    return ToolExecutionResult(
        tool_name="get_agent_by_house_id",
        ok=True,
        result={
            "house_id": request.house_id,
            "agent": {
                "agent_id": f"agent_for_{request.house_id}",
                "agent_name": "Mock Agent",
            },
            "source": "mock_data",
        },
    )
