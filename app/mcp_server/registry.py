"""Tool manifest loader and handler registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from app.common.settings import get_settings, resolve_from_root
from app.mcp_server.tool_schemas import ToolExecutionResult, ToolManifest, ToolManifestEntry
from app.mcp_server.tools.agent_tools import (
    get_agent_by_house_id,
    get_agent_id_by_name,
    get_houses_by_agent_id,
)
from app.mcp_server.tools.house_tools import (
    get_house_detail,
    update_house_name,
    update_house_price,
)
from app.mcp_server.tools.meta_tools import get_tool_detail, list_tools


ToolHandler = Callable[[dict[str, Any]], ToolExecutionResult]


class ToolRegistry:
    """Registry that binds manifest entries to Python handlers."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self._handlers: dict[str, ToolHandler] = {
            "list_tools": lambda payload: list_tools(self.manifest, payload),
            "get_tool_detail": lambda payload: get_tool_detail(self.manifest, payload),
            "get_agent_id_by_name": get_agent_id_by_name,
            "get_houses_by_agent_id": get_houses_by_agent_id,
            "get_agent_by_house_id": get_agent_by_house_id,
            "get_house_detail": get_house_detail,
            "update_house_name": update_house_name,
            "update_house_price": update_house_price,
        }

    def _load_manifest(self) -> ToolManifest:
        """Load the tool manifest from YAML."""
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return ToolManifest(**raw)

    def list_entries(self) -> list[ToolManifestEntry]:
        """Return all registered tool manifest entries."""
        return self.manifest.tools

    def get_entry(self, tool_name: str) -> ToolManifestEntry | None:
        """Return one tool manifest entry by name."""
        return next((item for item in self.manifest.tools if item.name == tool_name), None)

    def invoke(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Execute one registered tool handler."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                ok=False,
                error=f"Unknown tool: {tool_name}",
            )
        return handler(arguments)


def build_registry() -> ToolRegistry:
    """Construct the tool registry from shared settings."""
    settings = get_settings()
    manifest_path = resolve_from_root(settings.mcp.manifest_path)
    return ToolRegistry(manifest_path=manifest_path)
