#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
PYTORCH_CHANNEL="pytorch"
NVIDIA_CHANNEL="nvidia"
CUDA_PACKAGE_VERSION="12.4"

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

ensure_gpu_runtime() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[bootstrap] Non-Linux platform detected, skipping CUDA runtime bootstrap."
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[bootstrap] nvidia-smi not found, skipping CUDA runtime bootstrap."
    return
  fi

  echo "[bootstrap] Checking PyTorch CUDA availability in conda env ${ENV_NAME}"
  if conda run --no-capture-output -n "$ENV_NAME" python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "[bootstrap] PyTorch CUDA is already available."
    return
  fi

  echo "[bootstrap] Resetting existing PyTorch packages in ${ENV_NAME}"
  conda remove -n "$ENV_NAME" -y pytorch torchvision torchaudio pytorch-cuda >/dev/null 2>&1 || true

  echo "[bootstrap] Installing CUDA-enabled PyTorch into ${ENV_NAME}"
  echo "[bootstrap] Command: conda install -n ${ENV_NAME} -y pytorch torchvision torchaudio pytorch-cuda=${CUDA_PACKAGE_VERSION} -c ${PYTORCH_CHANNEL} -c ${NVIDIA_CHANNEL}"
  conda install -n "$ENV_NAME" -y \
    pytorch torchvision torchaudio "pytorch-cuda=${CUDA_PACKAGE_VERSION}" \
    -c "$PYTORCH_CHANNEL" -c "$NVIDIA_CHANNEL"

  echo "[bootstrap] Verifying PyTorch CUDA availability"
  conda run --no-capture-output -n "$ENV_NAME" python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
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

ensure_gpu_runtime

cd "$ROOT_DIR"
echo "[bootstrap] Running init.py inside conda env ${ENV_NAME}"
echo "[bootstrap] Command: conda run --no-capture-output -n ${ENV_NAME} python -u init.py $*"
exec conda run --no-capture-output -n "$ENV_NAME" python -u init.py "$@"
