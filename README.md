# Local LangGraph Tool-Use Service

This project is a local-first starter scaffold for a production-style tool-use system with three services:

- An OpenAI-compatible LLM service that serves a local Qwen model.
- An MCP-style internal tool server that exposes explicit tool contracts over FastAPI.
- A FastAPI gateway that runs a LangGraph workflow and calls the MCP server as its tool layer.

## Architecture

```text
Client
  |
  v
Gateway / LangGraph (FastAPI, port 8000)
  |
  +--> MCP Tool Server (FastAPI, port 8002)
  |
  +--> Local LLM Service (OpenAI-compatible, port 8001)
```

### Service roles

- LLM service: hosts the local model behind an OpenAI-compatible `/v1/chat/completions` API.
- MCP server: owns tool manifests, schemas, request validation, and placeholder implementations.
- Gateway: receives user chat requests, runs the LangGraph flow, plans tool calls, executes them, reviews results, and returns the final answer.

## Directory Structure

```text
.
├── README.md
├── init.py
├── config
│   ├── config.yaml
│   └── tools.yaml
├── scripts
│   ├── start_all.sh
│   ├── start_gateway.sh
│   ├── start_llm.sh
│   └── start_mcp.sh
├── app
│   ├── common
│   │   ├── logging.py
│   │   ├── schemas.py
│   │   └── settings.py
│   ├── gateway
│   │   ├── api.py
│   │   ├── main.py
│   │   └── response_models.py
│   ├── graph
│   │   ├── build_graph.py
│   │   ├── nodes.py
│   │   ├── prompts.py
│   │   ├── router.py
│   │   └── state.py
│   ├── llm_client
│   │   └── openai_compatible.py
│   └── mcp_server
│       ├── docs
│       │   └── TOOL_SPEC.md
│       ├── main.py
│       ├── registry.py
│       ├── tool_schemas.py
│       └── tools
│           ├── agent_tools.py
│           ├── house_tools.py
│           └── meta_tools.py
└── logs
```

## Startup Order

Start the services in this order:

1. LLM service
2. MCP server
3. Gateway

You can start them separately:

```bash
bash scripts/start_llm.sh
bash scripts/start_mcp.sh
bash scripts/start_gateway.sh
```

Or together:

```bash
bash scripts/start_all.sh
```

## Configuration

All service configuration lives in `config/config.yaml`.

### Main sections

- `project`: environment name and Python version used by `init.py`.
- `llm`: model name, OpenAI-compatible base URL, default generation settings, and local serving settings.
- `mcp`: MCP server binding, manifest file path, and request timeout.
- `gateway`: gateway binding, debug flags, and in-memory trace store size.
- `logging`: log level and optional JSON logging.

### Tool manifest

`config/tools.yaml` stores:

- Tool name
- Short purpose description
- Input fields
- Output fields
- Mode (`read` or `write`)
- Risk level
- Future extension notes

## Default Model

The default model config uses the official Qwen model:

```yaml
llm:
  model_name: Qwen/Qwen3-1.7B
  service:
    model_source: Qwen/Qwen3-1.7B
```

## How To Swap Models

To swap the model later, only change `config/config.yaml`:

1. Update `llm.model_name`.
2. Update `llm.service.model_source`.
3. If needed, adjust `tensor_parallel_size`, `dtype`, or `base_url`.

No application code changes should be required if the new backend is still OpenAI-compatible.

## LangGraph Design

The gateway builds a minimal but extensible graph with these nodes:

1. `normalize_input`
2. `classify_intent`
3. `plan_tool_calls`
4. `execute_tools`
5. `review_results`
6. `finalize`

### How it works

- `normalize_input` cleans the latest user message.
- `classify_intent` asks the local model to label the request as tool lookup, read, write, or general.
- `plan_tool_calls` fetches the MCP tool manifest and asks the model which tools should be called.
- `execute_tools` sends the planned calls to the MCP server.
- `review_results` annotates mock data and tool failures.
- `finalize` asks the model to write the final answer using the reviewed tool output.

## Gateway API

The gateway exposes:

- `POST /v1/chat`
- `GET /health`
- `GET /v1/tools`
- `GET /v1/traces/{trace_id}`

## How LangGraph Interacts With MCP

The gateway does not import tool business logic directly. Instead:

1. It reads the user request.
2. It plans tool calls using the tool manifest served by the MCP server.
3. It invokes tools through the MCP server HTTP API.
4. It reviews the results and returns an answer.

This keeps the orchestration layer separate from tool execution, which makes later replacement of mock tools safer.

## How To Add A New Tool

1. Add the tool entry to `config/tools.yaml`.
2. Add request schema models in `app/mcp_server/tool_schemas.py`.
3. Implement the handler in `app/mcp_server/tools/`.
4. Register the handler in `app/mcp_server/registry.py`.
5. Update prompt wording in `app/graph/prompts.py` if the new tool changes planning behavior.

## Replacing Mock Tool Logic With Real Data Access

Right now the business-facing tools return mock responses marked with `source: mock_data` or `write_status: mock_not_persisted`.

To replace them with real database code later:

1. Keep the input and output schemas stable.
2. Replace the placeholder logic inside the tool functions with repository or service calls.
3. Keep validation in the schema layer.
4. Preserve the same handler registration path so the gateway does not need to change.

This makes it possible to move from local mocks to real persistence without redesigning the graph.

## Initialization

`init.py` can:

- Create a new conda environment
- Install required packages
- Ensure the expected project folders exist
- Print model preparation commands for the configured Qwen model

Example:

```bash
python init.py
python init.py --skip-conda
python init.py --skip-install
```

## Notes

- Python 3.11+ is assumed.
- This scaffold intentionally avoids Docker, Kubernetes, Redis, Celery, authentication, and real database integration.
- The write tools are placeholders and do not persist changes yet.
