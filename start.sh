#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[start] conda 未安装或不在 PATH 中。"
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

ENV_NAME="$(read_project_value environment_name)"

if [[ -z "${ENV_NAME}" ]]; then
  echo "[start] 无法从 config/config.yaml 读取 conda 环境配置。"
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[start] conda 环境不存在，请先执行 ./bootstrap.sh"
  exit 1
fi

cd "$ROOT_DIR"
exec conda run -n "$ENV_NAME" python start.py "$@"
