#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[bootstrap] conda was not found in PATH."
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
PYTHON_VERSION="$(read_project_value python_version)"

if [[ -z "${ENV_NAME}" || -z "${PYTHON_VERSION}" ]]; then
  echo "[bootstrap] Failed to read conda environment config from config/config.yaml."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[bootstrap] Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
else
  echo "[bootstrap] Using existing conda env: ${ENV_NAME}"
fi

cd "$ROOT_DIR"
echo "[bootstrap] Running init.py inside conda env ${ENV_NAME}"
echo "[bootstrap] Command: conda run --no-capture-output -n ${ENV_NAME} python -u init.py $*"
exec conda run --no-capture-output -n "$ENV_NAME" python -u init.py "$@"
