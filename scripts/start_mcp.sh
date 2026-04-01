#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec python - <<'PY'
from app.common.settings import get_settings
import subprocess
import sys

settings = get_settings()
raise SystemExit(
    subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.mcp_server.main:app",
            "--host",
            settings.mcp.service.host,
            "--port",
            str(settings.mcp.service.port),
        ]
    )
)
PY
