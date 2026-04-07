#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/config.yaml"
PYTORCH_CHANNEL="${PYTORCH_CHANNEL:-pytorch}"
NVIDIA_CHANNEL="${NVIDIA_CHANNEL:-nvidia}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
PYTORCH_CUDA_VERSION="${PYTORCH_CUDA_VERSION:-12.8}"
PYTORCH_WHEEL_INDEX_URL="${PYTORCH_WHEEL_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
DEFAULT_PIP_INDEX_URL="${DEFAULT_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
DEFAULT_HF_ENDPOINT="${DEFAULT_HF_ENDPOINT:-https://hf-mirror.com}"
DEFAULT_HF_TIMEOUT_SECONDS="${DEFAULT_HF_TIMEOUT_SECONDS:-60}"

log() {
  echo "[bootstrap] $*"
}

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

read_llm_service_value() {
  local key="$1"
  awk -v target="$key" '
    $0 ~ /^llm:[[:space:]]*$/ { in_llm=1; next }
    in_llm && $0 ~ /^[^[:space:]]/ { in_llm=0 }
    in_llm && $0 ~ /^[[:space:]]+service:[[:space:]]*$/ { in_service=1; next }
    in_service && $0 ~ /^[[:space:]]{2}[^[:space:]]/ && $1 != "service:" { in_service=0 }
    in_service {
      gsub(/"/, "", $0)
      if ($1 == target ":") {
        print $2
        exit
      }
    }
  ' "$CONFIG_PATH"
}

ensure_conda_available() {
  if ! command -v conda >/dev/null 2>&1; then
    log "conda was not found in PATH."
    exit 1
  fi
}

remove_pip_torch_packages() {
  log "Removing pip-installed torch packages from conda env ${ENV_NAME} if present"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
}

remove_torch_residuals() {
  log "Removing residual torch package directories from conda env ${ENV_NAME} if present"
  conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import shutil
import site
from pathlib import Path

targets = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchgen",
]

patterns = [
    "torch-*.dist-info",
    "torchvision-*.dist-info",
    "torchaudio-*.dist-info",
]

seen = set()
for base in [Path(path) for path in site.getsitepackages()]:
    if not base.exists():
        continue
    for name in targets:
        target = base / name
        if target.exists() and target not in seen:
            print(f"[bootstrap] removing residual path: {target}")
            shutil.rmtree(target, ignore_errors=True)
            seen.add(target)
    for pattern in patterns:
        for target in base.glob(pattern):
            if target.exists() and target not in seen:
                print(f"[bootstrap] removing residual path: {target}")
                shutil.rmtree(target, ignore_errors=True)
                seen.add(target)
PY
}

remove_conda_torch_packages() {
  log "Removing existing conda PyTorch packages from conda env ${ENV_NAME} if present"
  conda remove -n "$ENV_NAME" -y pytorch torchvision torchaudio pytorch-cuda >/dev/null 2>&1 || true
}

print_torch_state() {
  local stage="$1"
  log "PyTorch runtime snapshot (${stage})"
  conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import importlib.util
import sys

print(f"[bootstrap] sys.executable={sys.executable}")
spec = importlib.util.find_spec("torch")
print(f"[bootstrap] torch.spec={'present' if spec else 'missing'}")
if spec is None:
    raise SystemExit(0)

try:
    import torch
except Exception as exc:
    print(f"[bootstrap] torch import failed: {exc}")
    raise SystemExit(0)

print(f"[bootstrap] torch.__version__={torch.__version__}")
print(f"[bootstrap] torch.version.cuda={torch.version.cuda}")
print(f"[bootstrap] torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"[bootstrap] torch.cuda.device_count()={torch.cuda.device_count()}")
PY
}

install_pip_gpu_torch() {
  log "Installing CUDA-enabled PyTorch wheel into conda env ${ENV_NAME}"
  log "Torch target versions: torch=${PYTORCH_VERSION}, torchvision=${TORCHVISION_VERSION}, torchaudio=${TORCHAUDIO_VERSION}, cuda=${PYTORCH_CUDA_VERSION}"
  log "PyTorch wheel index-url: ${PYTORCH_WHEEL_INDEX_URL}"
  log "Command: conda run --no-capture-output -n ${ENV_NAME} python -m pip install --upgrade torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url ${PYTORCH_WHEEL_INDEX_URL}"
  conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$PYTORCH_WHEEL_INDEX_URL"
}

ensure_gpu_runtime() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    log "Non-Linux platform detected, skipping CUDA runtime bootstrap."
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found, skipping CUDA runtime bootstrap."
    return
  fi

  log "Linux + NVIDIA detected. PyTorch install mode: pip wheel inside target conda env"
  log "Target conda env: ${ENV_NAME}"
  print_torch_state "before cleanup"
  remove_pip_torch_packages
  remove_torch_residuals
  remove_conda_torch_packages
  install_pip_gpu_torch
  print_torch_state "after pip wheel install"
}

ensure_conda_available

ENV_NAME="$(read_project_value environment_name)"
PYTHON_VERSION="$(read_project_value python_version)"
MODEL_SOURCE="$(read_llm_service_value model_source)"

if [[ -z "${ENV_NAME}" || -z "${PYTHON_VERSION}" ]]; then
  log "Failed to read conda environment config from config/config.yaml."
  exit 1
fi

log "Project root: ${ROOT_DIR}"
log "Target conda env: ${ENV_NAME}"
log "Requested Python version: ${PYTHON_VERSION}"

export PIP_INDEX_URL="${PIP_INDEX_URL:-$DEFAULT_PIP_INDEX_URL}"
export HF_ENDPOINT="${HF_ENDPOINT:-$DEFAULT_HF_ENDPOINT}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-$DEFAULT_HF_TIMEOUT_SECONDS}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-$DEFAULT_HF_TIMEOUT_SECONDS}"

log "Default pip index-url: ${PIP_INDEX_URL}"
log "Default Hugging Face endpoint: ${HF_ENDPOINT}"
log "Default Hugging Face timeout: ${HF_HUB_DOWNLOAD_TIMEOUT}s"
log "Configured model source: ${MODEL_SOURCE:-<unknown>}"
log "Model runtime note: after switching model family, run bootstrap.sh once without --skip-install so transformers runtime packages can be upgraded inside the Linux conda env."
log "Qwen compatibility note: init.py now detects Qwen3.5-style model_type values such as qwen3_5 from the selected model config and may install Transformers main branch automatically."

case "${MODEL_SOURCE,,}" in
  *qwen3.5*|*qwen3_5*|*qwen3-5*)
    export INIT_FORCE_RUNTIME_SYNC=1
    log "Detected Qwen3.5 model family. Forced runtime package sync is enabled for this bootstrap run."
    ;;
  *)
    ;;
esac

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  log "Creating conda env: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
else
  log "Using existing conda env: ${ENV_NAME}"
fi

ensure_gpu_runtime

cd "$ROOT_DIR"
log "Running init.py inside conda env ${ENV_NAME}"
log "Command: conda run --no-capture-output -n ${ENV_NAME} python -u init.py $*"
exec conda run --no-capture-output -n "$ENV_NAME" python -u init.py "$@"
