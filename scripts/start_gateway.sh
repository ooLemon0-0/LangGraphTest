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
            "app.gateway.main:app",
            "--host",
            settings.gateway.service.host,
            "--port",
            str(settings.gateway.service.port),
        ]
    )
)
PY
