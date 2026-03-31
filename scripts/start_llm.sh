#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"

read_config() {
  python - "$CONFIG_PATH" "$1" <<'PY'
import sys
import yaml

config_path = sys.argv[1]
field = sys.argv[2]

with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

parts = field.split(".")
value = config
for part in parts:
    value = value[part]
print(value)
PY
}

HOST="$(read_config llm.service.host)"
PORT="$(read_config llm.service.port)"
MODEL_SOURCE="$(read_config llm.service.model_source)"
DTYPE="$(read_config llm.service.dtype)"
TENSOR_PARALLEL_SIZE="$(read_config llm.service.tensor_parallel_size)"

exec python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_SOURCE" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
