#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

HOST="$(python - <<'PY'
from app.common.settings import get_settings
print(get_settings().mcp.service.host)
PY
)"

PORT="$(python - <<'PY'
from app.common.settings import get_settings
print(get_settings().mcp.service.port)
PY
)"

exec python -m uvicorn app.mcp_server.main:app --host "$HOST" --port "$PORT"
