"""Placeholder house-related MCP tools."""

from __future__ import annotations

from app.mcp_server.tool_schemas import (
    GetHouseDetailInput,
    ToolExecutionResult,
    UpdateHouseNameInput,
    UpdateHousePriceInput,
)
from app.mcp_server.tools.mock_store import (
    get_house_detail_record,
    update_house_name_record,
    update_house_price_record,
)


def get_house_detail(payload: dict[str, object]) -> ToolExecutionResult:
    """Return a mock house record."""
    request = GetHouseDetailInput(**payload)
    house = get_house_detail_record(request.house_id)
    return ToolExecutionResult(
        tool_name="get_house_detail",
        ok=True,
        result={
            "house": house,
            "source": "mock_data",
        },
    )


def update_house_name(payload: dict[str, object]) -> ToolExecutionResult:
    """Return a mock write response for renaming a house."""
    request = UpdateHouseNameInput(**payload)
    update_house_name_record(request.house_id, request.new_name)
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
    update_house_price_record(request.house_id, request.new_price, request.currency)
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
