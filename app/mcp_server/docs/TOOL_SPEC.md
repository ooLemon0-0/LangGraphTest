# MCP Tool Spec

This server exposes a small internal HTTP API that acts as the MCP-style tool layer for the LangGraph gateway.

## Goals

- Keep tool contracts explicit and simple.
- Make each tool easy for a small model to choose.
- Separate tool schemas from tool implementations.
- Allow later replacement of mock logic with real data access code.

## Internal API

### `GET /tools`

Returns every tool from `config/tools.yaml`.

Sample response:

```json
{
  "tools": [
    {
      "name": "get_house_detail",
      "description": "Get the full detail for one house id. Use this when you need the current name, price, status, and linked agent id.",
      "input_fields": ["house_id"],
      "output_fields": ["house", "source"],
      "mode": "read",
      "risk_level": "low",
      "future_extension_notes": "Add richer nested fields and permission-aware projections."
    }
  ]
}
```

### `GET /tools/{tool_name}`

Returns one full manifest entry.

### `POST /invoke`

Request body:

```json
{
  "tool_name": "get_house_detail",
  "arguments": {
    "house_id": "house_123"
  },
  "trace_id": "optional-trace-id"
}
```

Response body:

```json
{
  "tool_name": "get_house_detail",
  "ok": true,
  "result": {
    "house": {
      "house_id": "house_123",
      "name": "Mock Harbor House",
      "price": 750000.0,
      "currency": "USD",
      "agent_id": "agent_demo_001",
      "status": "mock_active"
    },
    "source": "mock_data"
  },
  "error": null,
  "mock": true
}
```

## Tool Rules

- Read tools should not mutate state.
- Write tools should only run when the user explicitly asks for a change.
- Mock implementations must clearly mark their output as mock data.
- Real persistence logic should be added behind the same function signatures.

## Extension Pattern

1. Add the tool metadata to `config/tools.yaml`.
2. Add input models to `app/mcp_server/tool_schemas.py`.
3. Implement the placeholder or real handler in `app/mcp_server/tools/`.
4. Register the handler in `app/mcp_server/registry.py`.
5. Update gateway prompting if the new tool changes planning behavior.
