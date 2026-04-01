"""Initialize project dependencies and model assets inside the active conda environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
DEFAULTS = {
    "llm.service.server_backend": "transformers",
    "llm.service.model_source": "Qwen/Qwen3-1.7B",
    "llm.service.model_source_cn": "Qwen/Qwen3-1.7B",
    "llm.service.model_cache_dir": "models",
    "llm.service.download_timeout": "60",
}
TORCH_REQUIREMENT_PREFIXES = ("torch", "torchvision", "torchaudio")


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Install dependencies and download the configured LLM model.")
    parser.add_argument("--skip-install", action="store_true", help="Skip Python package installation.")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download.")
    parser.add_argument("--force", action="store_true", help="Force dependency upgrades while installing.")
    return parser.parse_args()


def ensure_directories() -> None:
    """Create runtime directories used by the project."""
    for directory in [ROOT_DIR / "logs", ROOT_DIR / "models"]:
        directory.mkdir(parents=True, exist_ok=True)


def load_config_values() -> dict[str, str]:
    """Load the minimal config values needed before importing project dependencies."""
    values = dict(DEFAULTS)
    if not CONFIG_PATH.exists():
        return values

    return parse_yaml_scalars(CONFIG_PATH, values)


def parse_yaml_scalars(path: Path, defaults: dict[str, str]) -> dict[str, str]:
    """Best-effort parser for the few scalar config keys needed during bootstrap."""
    values = dict(defaults)
    stack: list[tuple[int, str]] = []
    section_pattern = re.compile(r"^([A-Za-z0-9_]+):\s*$")
    scalar_pattern = re.compile(r"^([A-Za-z0-9_]+):\s*(.+?)\s*$")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        while stack and stack[-1][0] >= indent:
            stack.pop()

        stripped = line.strip()
        section_match = section_pattern.match(stripped)
        if section_match:
            stack.append((indent, section_match.group(1)))
            continue

        scalar_match = scalar_pattern.match(stripped)
        if not scalar_match:
            continue

        key, value = scalar_match.groups()
        path_key = ".".join([part for _, part in stack] + [key])
        if path_key not in values:
            continue

        values[path_key] = value.strip().strip("'\"")

    return values


def current_conda_env() -> str:
    """Return the active conda environment name if present."""
    return os.environ.get("CONDA_DEFAULT_ENV", "").strip()


def run_command(command: list[str], description: str) -> None:
    """Run one subprocess and stop on failure."""
    print(f"[init] {description}")
    print("       " + " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def command_exists(name: str) -> bool:
    """Return True when a command is available on PATH."""
    return shutil.which(name) is not None


def effective_backend(config: dict[str, str]) -> str:
    """Resolve the backend actually used on the current OS."""
    backend = config["llm.service.server_backend"].strip().lower()
    if backend == "vllm" and os.name == "nt":
        print("[init] Windows detected, automatically switching vLLM to transformers.")
        return "transformers"
    if backend == "vllm" and sys.platform == "darwin":
        print("[init] macOS detected, automatically switching vLLM to transformers.")
        return "transformers"
    return backend


def install_packages(force: bool, backend: str) -> None:
    """Install project dependencies in the current interpreter environment."""
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")

    install_torch_runtime(force=force)
    install_python_requirements(force=force)

    if backend == "vllm":
        vllm_command = [sys.executable, "-m", "pip", "install"]
        if force:
            vllm_command.append("--upgrade")
        vllm_command.append("vllm")
        run_command(vllm_command, "Installing optional vLLM backend")


def install_python_requirements(force: bool) -> None:
    """Install project Python requirements excluding torch-family packages."""
    filtered_requirements = build_filtered_requirements_file()
    try:
        install_command = [sys.executable, "-m", "pip", "install"]
        if force:
            install_command.append("--upgrade")
        install_command.extend(["-r", str(filtered_requirements)])
        run_command(install_command, "Installing project Python requirements")
    finally:
        filtered_requirements.unlink(missing_ok=True)


def build_filtered_requirements_file() -> Path:
    """Create a temporary requirements file without torch-family packages."""
    source = ROOT_DIR / "requirements.txt"
    if not source.exists():
        raise FileNotFoundError(f"requirements.txt was not found: {source}")

    filtered_lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            filtered_lines.append(raw_line)
            continue
        normalized = stripped.lower()
        if normalized.startswith(TORCH_REQUIREMENT_PREFIXES):
            continue
        filtered_lines.append(raw_line)

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="requirements.filtered.",
        suffix=".txt",
        dir=str(ROOT_DIR),
        delete=False,
    )
    with handle:
        handle.write("\n".join(filtered_lines) + "\n")
    return Path(handle.name)


def install_torch_runtime(force: bool) -> None:
    """Install an appropriate torch runtime without mixing pip and conda variants."""
    env_name = current_conda_env()
    if sys.platform.startswith("linux") and command_exists("nvidia-smi") and env_name:
        install_conda_torch(env_name=env_name, force=force)
        return

    pip_command = [sys.executable, "-m", "pip", "install"]
    if force:
        pip_command.append("--upgrade")
    pip_command.extend(["torch", "torchvision", "torchaudio"])
    run_command(pip_command, "Installing torch runtime with pip")


def install_conda_torch(env_name: str, force: bool) -> None:
    """Install CUDA-enabled torch via conda for Linux NVIDIA hosts."""
    verify_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        "-c",
        "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
    ]
    verified = subprocess.run(
        verify_command,
        cwd=ROOT_DIR,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if verified.returncode == 0 and not force:
        print(f"[init] CUDA-enabled PyTorch already available in conda env {env_name}.")
        return

    remove_command = [
        "conda",
        "remove",
        "-n",
        env_name,
        "-y",
        "pytorch",
        "torchvision",
        "torchaudio",
        "pytorch-cuda",
    ]
    print(f"[init] Resetting torch runtime in conda env {env_name}.")
    subprocess.run(remove_command, cwd=ROOT_DIR, check=False)

    install_command = [
        "conda",
        "install",
        "-n",
        env_name,
        "-y",
        "pytorch",
        "torchvision",
        "torchaudio",
        "pytorch-cuda=12.4",
        "-c",
        "pytorch",
        "-c",
        "nvidia",
    ]
    run_command(install_command, "Installing CUDA-enabled PyTorch with conda")


def safe_rmtree(path: Path) -> None:
    """Remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=False)


def is_network_timeout_error(exc: Exception) -> bool:
    """Return True when an exception looks like a transient network or timeout failure."""
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError)):
        return True

    message = " ".join(
        [
            exc.__class__.__name__,
            exc.__class__.__module__,
            str(exc),
            repr(exc),
        ]
    ).lower()

    markers = [
        "timeout",
        "timed out",
        "read timed out",
        "connect timeout",
        "connection aborted",
        "connection reset",
        "connection refused",
        "temporary failure in name resolution",
        "name or service not known",
        "max retries exceeded",
        "remote disconnected",
        "network is unreachable",
        "proxyerror",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "503",
        "502",
        "504",
        "connectionerror",
        "readtimeout",
        "connecttimeout",
    ]
    return any(marker in message for marker in markers)


def ensure_modelscope_installed(force: bool = False) -> None:
    """Install ModelScope only when needed, or upgrade it when forced."""
    if not force and importlib.util.find_spec("modelscope") is not None:
        return

    command = [sys.executable, "-m", "pip", "install"]
    if force:
        command.append("--upgrade")
    command.append("modelscope")
    run_command(command, "Installing ModelScope")


def finalize_download(staging_dir: Path, target_dir: Path) -> Path:
    """Replace the target model directory with a fully downloaded staging directory."""
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        safe_rmtree(target_dir)
    shutil.move(str(staging_dir), str(target_dir))
    return target_dir


def model_name_from_source(model_source: str) -> str:
    """Extract a stable local directory name from a repo id."""
    normalized = model_source.strip().rstrip("/")
    return normalized.split("/")[-1]


def build_staging_dir(target_dir: Path) -> Path:
    """Create a unique staging directory near the final model path."""
    staging_dir = target_dir.parent / f".{target_dir.name}.staging-{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def set_huggingface_timeout_env(timeout_seconds: int) -> dict[str, str | None]:
    """Apply temporary Hugging Face timeout env vars and return prior values."""
    previous = {
        "HF_HUB_DOWNLOAD_TIMEOUT": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT"),
        "HF_HUB_ETAG_TIMEOUT": os.environ.get("HF_HUB_ETAG_TIMEOUT"),
    }
    timeout_value = str(timeout_seconds)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = timeout_value
    os.environ["HF_HUB_ETAG_TIMEOUT"] = timeout_value
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    """Restore environment variables modified during download."""
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def download_from_modelscope(model_source_cn: str, staging_dir: Path) -> Path:
    """Download a model from ModelScope into the staging directory."""
    print(f"[init] Downloading from ModelScope: {model_source_cn}")
    from modelscope import snapshot_download  # type: ignore

    source_cache_dir = staging_dir.parent / f"{staging_dir.name}-modelscope-cache"
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = Path(
            snapshot_download(
                model_id=model_source_cn,
                cache_dir=str(source_cache_dir),
            )
        ).resolve()
        if not downloaded_path.exists():
            raise RuntimeError(f"ModelScope did not produce a local model directory: {downloaded_path}")

        if any(staging_dir.iterdir()):
            safe_rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(downloaded_path, staging_dir, dirs_exist_ok=True)
        return staging_dir
    finally:
        safe_rmtree(source_cache_dir)


def download_from_huggingface(model_source: str, staging_dir: Path, timeout_seconds: int) -> Path:
    """Download a model from Hugging Face into the staging directory."""
    print(f"[init] Downloading from Hugging Face: {model_source}")
    from huggingface_hub import snapshot_download

    previous_env = set_huggingface_timeout_env(timeout_seconds)
    try:
        snapshot_download(
            repo_id=model_source,
            local_dir=str(staging_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    finally:
        restore_env(previous_env)
    return staging_dir


def download_model(
    model_source: str,
    model_source_cn: str,
    model_cache_dir: str,
    timeout_seconds: int,
    force: bool,
) -> Path:
    """Download the configured model with ModelScope-first fallback behavior."""
    model_name = model_name_from_source(model_source_cn or model_source)
    target_dir = (ROOT_DIR / model_cache_dir / model_name).resolve()
    staging_dir = build_staging_dir(target_dir)

    try:
        try:
            ensure_modelscope_installed(force=force)
            download_from_modelscope(model_source_cn=model_source_cn, staging_dir=staging_dir)
            result = finalize_download(staging_dir=staging_dir, target_dir=target_dir)
            print(f"[init] Model download complete via ModelScope: {result}")
            return result
        except Exception as exc:
            safe_rmtree(staging_dir)
            staging_dir = build_staging_dir(target_dir)
            if not is_network_timeout_error(exc):
                raise

            print(f"[init] ModelScope failed, falling back to Hugging Face: {exc}")
            download_from_huggingface(
                model_source=model_source,
                staging_dir=staging_dir,
                timeout_seconds=timeout_seconds,
            )
            result = finalize_download(staging_dir=staging_dir, target_dir=target_dir)
            print(f"[init] Model download complete via Hugging Face: {result}")
            return result
    except Exception:
        safe_rmtree(staging_dir)
        raise


def main() -> None:
    """Run dependency installation and model preparation."""
    args = parse_args()
    config = load_config_values()
    backend = effective_backend(config)
    env_name = current_conda_env()

    ensure_directories()

    if not env_name:
        print("[init] Warning: no active conda environment detected. This script is expected to run via bootstrap.")
    else:
        print(f"[init] Active conda environment: {env_name}")

    if not args.skip_install:
        install_packages(force=args.force, backend=backend)

    if not args.skip_model:
        download_model(
            model_source=config["llm.service.model_source"],
            model_source_cn=config["llm.service.model_source_cn"],
            model_cache_dir=config["llm.service.model_cache_dir"],
            timeout_seconds=int(config["llm.service.download_timeout"]),
            force=args.force,
        )

    print()
    print("[init] Initialization complete.")
    print(f"[init] Backend: {backend}")
    print("[init] Next step: run the platform start launcher.")


if __name__ == "__main__":
    main()
