"""FastAPI entrypoint for the local MCP tool server."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.common.logging import configure_logging
from app.common.schemas import HealthResponse, ToolInvocationRequest, ToolInvocationResponse
from app.common.settings import get_settings
from app.mcp_server.registry import build_registry


settings = get_settings()
configure_logging(settings.logging.level, settings.logging.json)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local MCP Tool Server", version="0.1.0")
registry = build_registry()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health."""
    return HealthResponse(status="ok", service="mcp_server")


@app.get("/tools")
async def list_registered_tools() -> dict[str, list[dict[str, object]]]:
    """Return the tool manifest contents."""
    return {"tools": [entry.model_dump() for entry in registry.list_entries()]}


@app.get("/tools/{tool_name}")
async def get_registered_tool(tool_name: str) -> dict[str, object]:
    """Return one tool manifest entry."""
    entry = registry.get_entry(tool_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    return {"tool": entry.model_dump()}


@app.post("/invoke", response_model=ToolInvocationResponse)
async def invoke_tool(request: ToolInvocationRequest) -> ToolInvocationResponse:
    """Invoke one tool through the registry."""
    logger.info("Invoking tool %s", request.tool_name)
    result = registry.invoke(request.tool_name, request.arguments)
    if not result.ok:
        raise HTTPException(status_code=400, detail=result.error)
    return ToolInvocationResponse(
        tool_name=result.tool_name,
        ok=result.ok,
        result=result.result,
        error=result.error,
        mock=result.mock,
    )
