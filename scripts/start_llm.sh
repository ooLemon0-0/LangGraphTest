#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec python - <<'PY'
from app.common.settings import get_settings
import os
import platform
import subprocess
import sys

settings = get_settings()
backend = settings.llm.service.server_backend.strip().lower()
if backend == "vllm" and (os.name == "nt" or platform.system().lower() == "darwin"):
    backend = "transformers"

if backend == "vllm":
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        settings.llm.service.host,
        "--port",
        str(settings.llm.service.port),
        "--model",
        settings.llm.service.model_source,
        "--dtype",
        settings.llm.service.dtype,
        "--tensor-parallel-size",
        str(settings.llm.service.tensor_parallel_size),
        "--download-dir",
        str((__import__("pathlib").Path.cwd() / settings.llm.service.model_cache_dir).resolve()),
    ]
else:
    command = [sys.executable, "-m", "app.llm_server.main"]

raise SystemExit(subprocess.call(command))
PY
