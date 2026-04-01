#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[start] conda was not found in PATH."
  exit 1
fi

read_project_value() {
  local key="$1"
  awk -v target="$key" '
    $0 ~ /^project:[[:space:]]*$/ { inside=1; next }
    inside && $0 ~ /^[^[:space:]]/ { inside=0 }
    inside {
      gsub(/"/, "", $0)
      if ($1 == target ":") {
        print $2
        exit
      }
    }
  ' "$CONFIG_PATH"
}

read_service_value() {
  local section="$1"
  local key="$2"
  awk -v target_section="$section" -v target_key="$key" '
    $0 ~ /^llm:[[:space:]]*$/ { top="llm"; next }
    $0 ~ /^mcp:[[:space:]]*$/ { top="mcp"; next }
    $0 ~ /^gateway:[[:space:]]*$/ { top="gateway"; next }
    $0 ~ /^[^[:space:]]/ && $0 !~ /:$/ { top="" }
    top == target_section && $0 ~ /^[[:space:]]+service:[[:space:]]*$/ { in_service=1; next }
    top != target_section { in_service=0 }
    in_service && $0 ~ /^[[:space:]]{4}[A-Za-z0-9_]+:/ {
      gsub(/"/, "", $0)
      split($0, parts, ":")
      key_name=parts[1]
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key_name)
      if (key_name == target_key) {
        value=substr($0, index($0, ":") + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
      }
    }
    in_service && $0 ~ /^[[:space:]]{2}[A-Za-z0-9_]+:/ && $0 !~ /^[[:space:]]{4}/ { in_service=0 }
  ' "$CONFIG_PATH"
}

ENV_NAME="$(read_project_value environment_name)"
GATEWAY_PORT="$(read_service_value gateway port)"
LLM_PORT="$(read_service_value llm port)"
MCP_PORT="$(read_service_value mcp port)"

if [[ -z "${ENV_NAME}" ]]; then
  echo "[start] Failed to read project.environment_name from config/config.yaml."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[start] Conda env ${ENV_NAME} was not found. Run ./bootstrap.sh first."
  exit 1
fi

cd "$ROOT_DIR"
echo "[start] Using conda env: ${ENV_NAME}"
echo "[start] Expected ports: gateway=${GATEWAY_PORT:-8000}, llm=${LLM_PORT:-8001}, mcp=${MCP_PORT:-8002}"
echo "[start] Logs directory: ${ROOT_DIR}/logs"
echo "[start] Command: conda run --no-capture-output -n ${ENV_NAME} python -u start.py $*"
exec conda run --no-capture-output -n "$ENV_NAME" python -u start.py "$@"
