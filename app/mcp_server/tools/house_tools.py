"""Placeholder house-related MCP tools."""

from __future__ import annotations

from app.mcp_server.tool_schemas import (
    GetHouseDetailInput,
    ToolExecutionResult,
    UpdateHouseNameInput,
    UpdateHousePriceInput,
)


def get_house_detail(payload: dict[str, object]) -> ToolExecutionResult:
    """Return a mock house record."""
    request = GetHouseDetailInput(**payload)
    return ToolExecutionResult(
        tool_name="get_house_detail",
        ok=True,
        result={
            "house": {
                "house_id": request.house_id,
                "name": "Mock Harbor House",
                "price": 750000.0,
                "currency": "USD",
                "agent_id": "agent_demo_001",
                "status": "mock_active",
            },
            "source": "mock_data",
        },
    )


def update_house_name(payload: dict[str, object]) -> ToolExecutionResult:
    """Return a mock write response for renaming a house."""
    request = UpdateHouseNameInput(**payload)
    return ToolExecutionResult(
        tool_name="update_house_name",
        ok=True,
        result={
            "house_id": request.house_id,
            "updated_name": request.new_name,
            "write_status": "mock_not_persisted",
            "source": "mock_data",
        },
    )


def update_house_price(payload: dict[str, object]) -> ToolExecutionResult:
    """Return a mock write response for updating a house price."""
    request = UpdateHousePriceInput(**payload)
    return ToolExecutionResult(
        tool_name="update_house_price",
        ok=True,
        result={
            "house_id": request.house_id,
            "updated_price": request.new_price,
            "currency": request.currency,
            "write_status": "mock_not_persisted",
            "source": "mock_data",
        },
    )
