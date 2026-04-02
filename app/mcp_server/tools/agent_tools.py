"""Placeholder agent-related MCP tools."""

from __future__ import annotations

from app.mcp_server.tool_schemas import (
    GetAgentByHouseIdInput,
    GetAgentIdByNameInput,
    GetHousesByAgentIdInput,
    ToolExecutionResult,
)
from app.mcp_server.tools.mock_store import get_house_detail_record, get_houses_for_agent, resolve_agent_by_name


def get_agent_id_by_name(payload: dict[str, object]) -> ToolExecutionResult:
    """Resolve a mock agent id from a name."""
    request = GetAgentIdByNameInput(**payload)
    agent = resolve_agent_by_name(request.agent_name)
    return ToolExecutionResult(
        tool_name="get_agent_id_by_name",
        ok=True,
        result={
            "agent_id": agent["agent_id"],
            "agent_name": agent["agent_name"],
            "source": "mock_data",
        },
    )


def get_houses_by_agent_id(payload: dict[str, object]) -> ToolExecutionResult:
    """Return mock houses for a given agent id."""
    request = GetHousesByAgentIdInput(**payload)
    houses = [
        {"house_id": house["house_id"], "name": house["name"]}
        for house in get_houses_for_agent(request.agent_id)
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
    house = get_house_detail_record(request.house_id)
    agent_id = str(house["agent_id"])
    return ToolExecutionResult(
        tool_name="get_agent_by_house_id",
        ok=True,
        result={
            "house_id": request.house_id,
            "agent": {
                "agent_id": agent_id,
                "agent_name": resolve_agent_by_name(agent_id.removeprefix("agent_"))["agent_name"]
                if agent_id.startswith("agent_")
                else "Mock Agent",
            },
            "source": "mock_data",
        },
    )
