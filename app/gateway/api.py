"""API routes for the FastAPI gateway."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from app.common.schemas import ChatRequest, ChatResponse, HealthResponse
from app.common.settings import AppSettings
from app.gateway.response_models import ToolManifestListResponse, TraceResponse
from app.graph.build_graph import build_graph


class TraceStore:
    """In-memory trace store for local development."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def put(self, trace_id: str, state: dict[str, Any]) -> None:
        """Store a trace snapshot with bounded size."""
        self._items[trace_id] = state
        self._items.move_to_end(trace_id)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)

    def get(self, trace_id: str) -> dict[str, Any] | None:
        """Return a stored trace."""
        return self._items.get(trace_id)


def create_router(settings: AppSettings) -> APIRouter:
    """Build the API router with bound dependencies."""
    router = APIRouter()
    graph = build_graph(settings)
    traces = TraceStore(limit=settings.gateway.trace_store_limit)

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return gateway health."""
        return HealthResponse(status="ok", service="gateway")

    @router.post("/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Run the LangGraph workflow for a chat request."""
        initial_state = {
            "trace_id": request.trace_id,
            "messages": request.messages,
        }
        result = await graph.ainvoke(initial_state)
        serializable_state = _serialize_state(result)
        traces.put(request.trace_id, serializable_state)
        return ChatResponse(
            trace_id=request.trace_id,
            answer=result.get("final_answer", ""),
            tool_calls=result.get("tool_plan", []),
            review_notes=result.get("review_notes", []),
            raw_state=serializable_state,
        )

    @router.get("/v1/tools", response_model=ToolManifestListResponse)
    async def list_tools() -> ToolManifestListResponse:
        """Proxy the MCP manifest for external callers."""
        mcp_service = settings.mcp.service
        async with httpx.AsyncClient(timeout=settings.mcp.request_timeout_seconds) as client:
            response = await client.get(f"http://{mcp_service.host}:{mcp_service.port}/tools")
            response.raise_for_status()
            data = response.json()
        return ToolManifestListResponse(**data)

    @router.get("/v1/traces/{trace_id}", response_model=TraceResponse)
    async def get_trace(trace_id: str) -> TraceResponse:
        """Return one stored trace."""
        trace = traces.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")
        return TraceResponse(trace_id=trace_id, state=trace)

    return router


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert graph state into JSON-friendly data."""
    serialized: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, list):
            serialized[key] = [item.model_dump() if hasattr(item, "model_dump") else item for item in value]
            continue
        serialized[key] = value.model_dump() if hasattr(value, "model_dump") else value
    return serialized
