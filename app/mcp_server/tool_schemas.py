"""Pydantic models for MCP tool definitions and payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolField(BaseModel):
    """One input or output field in the tool manifest."""

    name: str


class ToolManifestEntry(BaseModel):
    """Tool metadata loaded from the YAML manifest."""

    name: str
    description: str
    description_zh: str = ""
    display_name_zh: str = ""
    aliases_zh: list[str] = Field(default_factory=list)
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    mode: Literal["read", "write"]
    risk_level: Literal["low", "medium", "high"]
    future_extension_notes: str
    tags: list[str] = Field(default_factory=list)
    business_domain: str = ""
    permission: str = ""


class ToolManifest(BaseModel):
    """Collection of tool entries."""

    tools: list[ToolManifestEntry] = Field(default_factory=list)


class ListToolsInput(BaseModel):
    """No-argument request for listing tools."""

    pass


class GetToolDetailInput(BaseModel):
    """Lookup a tool by name."""

    tool_name: str = Field(description="The exact tool name from the tool manifest.")


class GetAgentIdByNameInput(BaseModel):
    """Resolve an agent id from an agent name."""

    agent_name: str = Field(description="Human-readable agent name.")


class GetHousesByAgentIdInput(BaseModel):
    """List houses assigned to an agent."""

    agent_id: str = Field(description="Unique agent identifier.")


class GetAgentByHouseIdInput(BaseModel):
    """Find the agent for one house."""

    house_id: str = Field(description="Unique house identifier.")


class GetHouseDetailInput(BaseModel):
    """Fetch detailed house information."""

    house_id: str = Field(description="Unique house identifier.")


class UpdateHouseNameInput(BaseModel):
    """Rename one house."""

    house_id: str = Field(description="Unique house identifier.")
    new_name: str = Field(description="New display name for the house.")


class UpdateHousePriceInput(BaseModel):
    """Update a house price."""

    house_id: str = Field(description="Unique house identifier.")
    new_price: float = Field(description="New price value.")
    currency: str = Field(default="USD", description="Currency code for the new price.")


class ToolExecutionResult(BaseModel):
    """Normalized result returned by the MCP server."""

    tool_name: str
    ok: bool
    mock: bool = True
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


TOOL_INPUT_MODELS = {
    "list_tools": ListToolsInput,
    "get_tool_detail": GetToolDetailInput,
    "get_agent_id_by_name": GetAgentIdByNameInput,
    "get_houses_by_agent_id": GetHousesByAgentIdInput,
    "get_agent_by_house_id": GetAgentByHouseIdInput,
    "get_house_detail": GetHouseDetailInput,
    "update_house_name": UpdateHouseNameInput,
    "update_house_price": UpdateHousePriceInput,
}
