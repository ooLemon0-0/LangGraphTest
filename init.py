"""Initialize project dependencies and model assets inside the active conda environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    "llm.service.model_source": "Qwen/Qwen3.5-4B",
    "llm.service.model_source_cn": "Qwen/Qwen3.5-4B",
    "llm.service.model_cache_dir": "models",
    "llm.service.download_timeout": "60",
    "retrieval.enabled": "true",
    "retrieval.model_name": "BAAI/bge-small-zh-v1.5",
    "retrieval.model_cache_dir": "models/embeddings",
}
TORCH_REQUIREMENT_PREFIXES = ("torch", "torchvision", "torchaudio")
DEFAULT_PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
OFFICIAL_PIP_INDEX_URL = "https://pypi.org/simple"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
MODEL_RUNTIME_PACKAGES = [
    "transformers>=4.57,<5.0",
    "tokenizers>=0.21,<1.0",
    "accelerate>=0.34,<1.0",
    "huggingface_hub>=0.30,<1.0",
    "safetensors>=0.4,<1.0",
    "sentencepiece>=0.2,<1.0",
]
TRANSFORMERS_MAIN_PACKAGE = "git+https://github.com/huggingface/transformers.git"
TRANSFORMERS_MAIN_MODEL_TYPES = {"qwen3_5"}
TRANSFORMERS_MAIN_SOURCE_HINTS = ("qwen3.5", "qwen3_5", "qwen3-5")
MODEL_RUNTIME_PACKAGE_NAMES = [
    "transformers",
    "tokenizers",
    "accelerate",
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Install dependencies and download the configured LLM model.")
    parser.add_argument("--skip-install", action="store_true", help="Skip Python package installation.")
    parser.add_argument("--skip-model", action="store_true", help="Skip model download.")
    parser.add_argument("--force", action="store_true", help="Force dependency upgrades while installing.")
    return parser.parse_args()


def log(message: str) -> None:
    """Print a consistently formatted init log line."""
    print(f"[init] {message}")


def ensure_directories() -> None:
    """Create runtime directories used by the project."""
    for directory in [ROOT_DIR / "logs", ROOT_DIR / "models", ROOT_DIR / "models" / "embeddings"]:
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


def command_exists(name: str) -> bool:
    """Return True when a command is available on PATH."""
    return shutil.which(name) is not None


def effective_backend(config: dict[str, str]) -> str:
    """Resolve the backend actually used on the current OS."""
    backend = config["llm.service.server_backend"].strip().lower()
    if backend == "vllm" and os.name == "nt":
        log("Windows detected, automatically switching vLLM to transformers.")
        return "transformers"
    if backend == "vllm" and sys.platform == "darwin":
        log("macOS detected, automatically switching vLLM to transformers.")
        return "transformers"
    return backend


def pip_index_url() -> str:
    """Return the preferred primary pip index URL."""
    return (
        os.environ.get("PIP_INDEX_URL")
        or os.environ.get("BOOTSTRAP_PIP_INDEX_URL")
        or DEFAULT_PIP_INDEX_URL
    )


def pip_extra_index_urls() -> list[str]:
    """Return configured extra pip index URLs."""
    raw = os.environ.get("PIP_EXTRA_INDEX_URL") or os.environ.get("BOOTSTRAP_PIP_EXTRA_INDEX_URL", "")
    return [value.strip() for value in raw.split() if value.strip()]


def run_command(command: list[str], description: str, env: dict[str, str] | None = None) -> None:
    """Run one subprocess and stop on failure."""
    log(description)
    print("       " + " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, check=True, env=env)


def build_pip_env(index_url: str, extra_index_urls: list[str] | None = None) -> dict[str, str]:
    """Create a pip environment override for one installation attempt."""
    env = os.environ.copy()
    env["PIP_INDEX_URL"] = index_url
    extras = extra_index_urls if extra_index_urls is not None else pip_extra_index_urls()
    if extras:
        env["PIP_EXTRA_INDEX_URL"] = " ".join(extras)
    else:
        env.pop("PIP_EXTRA_INDEX_URL", None)
    return env


def build_pip_command(
    *packages: str,
    upgrade: bool = False,
    force_reinstall: bool = False,
    no_cache_dir: bool = False,
) -> list[str]:
    """Construct a reusable pip install command."""
    command = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        command.append("--upgrade")
    if force_reinstall:
        command.append("--force-reinstall")
    if no_cache_dir:
        command.append("--no-cache-dir")
    command.extend(packages)
    return command


def run_pip_install(
    *packages: str,
    description: str,
    upgrade: bool = False,
    force_reinstall: bool = False,
    no_cache_dir: bool = False,
) -> None:
    """Install Python packages with domestic mirror first, then fall back to the official index."""
    attempts = [
        ("preferred", pip_index_url(), pip_extra_index_urls()),
        ("official", OFFICIAL_PIP_INDEX_URL, []),
    ]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    command = build_pip_command(
        *packages,
        upgrade=upgrade,
        force_reinstall=force_reinstall,
        no_cache_dir=no_cache_dir,
    )
    last_error: subprocess.CalledProcessError | None = None

    for label, index_url, extra_urls in attempts:
        key = (index_url, tuple(extra_urls))
        if key in seen:
            continue
        seen.add(key)

        env = build_pip_env(index_url=index_url, extra_index_urls=extra_urls)
        log(f"{description} via pip ({label} index)")
        log(f"Python executable: {sys.executable}")
        log(f"pip index-url: {env['PIP_INDEX_URL']}")
        if env.get("PIP_EXTRA_INDEX_URL"):
            log(f"pip extra-index-url: {env['PIP_EXTRA_INDEX_URL']}")
        else:
            log("pip extra-index-url: <none>")
        print("       " + " ".join(command))

        try:
            subprocess.run(command, cwd=ROOT_DIR, check=True, env=env)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            log(f"{description} failed via {label} index with exit code {exc.returncode}")

    if last_error is not None:
        raise last_error


def uninstall_python_packages(*packages: str, description: str) -> None:
    """Uninstall Python packages from the active interpreter if they are present."""
    if not packages:
        return

    command = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    log(description)
    print("       " + " ".join(command))
    subprocess.run(command, cwd=ROOT_DIR, check=False, env=os.environ.copy())


def detect_torch_installation_source() -> str:
    """Best-effort classification of how torch is installed in the active interpreter."""
    if importlib.util.find_spec("torch") is None:
        return "missing"

    torch_module = __import__("torch")
    location = Path(getattr(torch_module, "__file__", "")).resolve()
    location_text = str(location).lower()
    if "site-packages" in location_text:
        return "pip-or-python-site-packages"
    if "conda" in location_text:
        return "conda"
    return "unknown"


def print_torch_runtime_snapshot(stage: str) -> None:
    """Log the current torch runtime state for observability."""
    source = detect_torch_installation_source()
    log(f"PyTorch installation mode ({stage}): {source}")
    if importlib.util.find_spec("torch") is None:
        log("PyTorch module import: missing")
        return

    import torch

    log(f"torch.__version__: {torch.__version__}")
    log(f"torch.version.cuda: {torch.version.cuda}")
    log(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    log(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    log(f"torch module path: {Path(torch.__file__).resolve()}")


def install_packages(force: bool, backend: str, config: dict[str, str]) -> None:
    """Install project dependencies in the current interpreter environment."""
    print_torch_runtime_snapshot(stage="before pip installs")
    run_pip_install("pip", description="Upgrading pip", upgrade=True)
    install_python_requirements(force=force)
    install_model_runtime_packages(force=force, config=config)
    print_runtime_package_versions()

    if backend == "vllm":
        run_pip_install("vllm", description="Installing optional vLLM backend", upgrade=force)

    print_torch_runtime_snapshot(stage="after pip installs")


def install_python_requirements(force: bool) -> None:
    """Install project Python requirements excluding torch-family packages."""
    filtered_requirements = build_filtered_requirements_file()
    try:
        run_pip_install("-r", str(filtered_requirements), description="Installing project Python requirements", upgrade=force)
    finally:
        filtered_requirements.unlink(missing_ok=True)


def install_model_runtime_packages(force: bool, config: dict[str, str]) -> None:
    """Install runtime packages required by the selected local model family.

    Keep this explicit instead of relying only on requirements.txt so model
    upgrades can harden runtime compatibility in one place. This is especially
    important when switching to newer Qwen families that need newer
    transformers/tokenizers support than older environments may have.
    """
    requires_transformers_main = model_requires_transformers_main(config)
    force_runtime_sync = requires_transformers_main or force or should_force_runtime_sync()

    if force_runtime_sync:
        log("Forcing model runtime package sync for the selected model family")
        uninstall_python_packages(
            *MODEL_RUNTIME_PACKAGE_NAMES,
            description="Removing model runtime packages before forced reinstall",
        )

    run_pip_install(
        *MODEL_RUNTIME_PACKAGES,
        description="Installing model runtime compatibility packages",
        upgrade=True,
        force_reinstall=force_runtime_sync,
        no_cache_dir=force_runtime_sync,
    )
    if requires_transformers_main:
        run_pip_install(
            TRANSFORMERS_MAIN_PACKAGE,
            description="Installing Transformers main branch for Qwen3.5 compatibility",
            upgrade=True,
            force_reinstall=True,
            no_cache_dir=True,
        )


def print_runtime_package_versions() -> None:
    """Log key model runtime package versions after installation."""
    packages = ["transformers", "tokenizers", "accelerate", "huggingface_hub"]
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "<unknown>")
            log(f"{package} version: {version}")
        except Exception as exc:
            log(f"{package} version check failed: {exc}")


def configured_model_local_dir(config: dict[str, str]) -> Path:
    """Return the expected local directory for the selected model."""
    model_source = config.get("llm.service.model_source_cn") or config.get("llm.service.model_source") or ""
    model_cache_dir = config.get("llm.service.model_cache_dir") or "models"
    return (ROOT_DIR / model_cache_dir / model_name_from_source(model_source)).resolve()


def read_local_model_type(config: dict[str, str]) -> str:
    """Read model_type from a downloaded local config.json when present."""
    config_path = configured_model_local_dir(config) / "config.json"
    if not config_path.exists():
        return ""

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"Skipping local model_type detection because config.json could not be read: {exc}")
        return ""

    model_type = str(payload.get("model_type", "")).strip().lower()
    if model_type:
        log(f"Detected local model_type from {config_path}: {model_type}")
    return model_type


def model_requires_transformers_main(config: dict[str, str]) -> bool:
    """Return True when the selected model family needs Transformers main.

    Qwen3.5 model artifacts currently advertise a dev/main Transformers
    baseline, so a stable PyPI release may still miss the `qwen3_5`
    architecture even when it is numerically newer than the documented dev tag.
    """
    sources = [
        (config.get("llm.service.model_source") or "").strip().lower(),
        (config.get("llm.service.model_source_cn") or "").strip().lower(),
    ]
    if any(hint in source for source in sources for hint in TRANSFORMERS_MAIN_SOURCE_HINTS):
        return True

    model_type = read_local_model_type(config)
    return model_type in TRANSFORMERS_MAIN_MODEL_TYPES


def should_force_runtime_sync() -> bool:
    """Return True when bootstrap requested a forced runtime reinstall."""
    raw = os.environ.get("INIT_FORCE_RUNTIME_SYNC", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_filtered_requirements_file() -> Path:
    """Create a temporary requirements file without torch-family packages."""
    source = ROOT_DIR / "requirements.txt"
    if not source.exists():
        raise FileNotFoundError(f"requirements.txt was not found: {source}")

    filtered_lines: list[str] = []
    removed_lines: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            filtered_lines.append(raw_line)
            continue
        normalized = stripped.lower()
        if normalized.startswith(TORCH_REQUIREMENT_PREFIXES):
            removed_lines.append(stripped)
            continue
        filtered_lines.append(raw_line)

    if removed_lines:
        log("Filtered torch-family packages from requirements.txt to avoid pip/conda mixing:")
        for line in removed_lines:
            log(f"  - {line}")
    else:
        log("No torch-family packages found in requirements.txt")

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
        log("ModelScope already installed")
        return

    run_pip_install("modelscope", description="Installing ModelScope", upgrade=force)


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
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
    }
    timeout_value = str(timeout_seconds)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = timeout_value
    os.environ["HF_HUB_ETAG_TIMEOUT"] = timeout_value
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    log(f"Hugging Face endpoint: {os.environ['HF_ENDPOINT']}")
    log(f"Hugging Face timeout: {timeout_value}s")
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
    log(f"Model download source: ModelScope ({model_source_cn})")
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
    log(f"Model download source: Hugging Face ({model_source})")
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
            log(f"Model download complete via ModelScope: {result}")
            return result
        except Exception as exc:
            safe_rmtree(staging_dir)
            staging_dir = build_staging_dir(target_dir)
            if not is_network_timeout_error(exc):
                raise

            log(f"ModelScope failed, falling back to Hugging Face: {exc}")
            download_from_huggingface(
                model_source=model_source,
                staging_dir=staging_dir,
                timeout_seconds=timeout_seconds,
            )
            result = finalize_download(staging_dir=staging_dir, target_dir=target_dir)
            log(f"Model download complete via Hugging Face: {result}")
            return result
    except Exception:
        safe_rmtree(staging_dir)
        raise


def download_embedding_model(
    model_name: str,
    model_cache_dir: str,
    timeout_seconds: int,
    force: bool,
) -> Path:
    """Download the retrieval embedding model ahead of service startup."""
    target_dir = (ROOT_DIR / model_cache_dir / model_name_from_source(model_name)).resolve()
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        log(f"Embedding model already present: {target_dir}")
        return target_dir

    staging_dir = build_staging_dir(target_dir)
    try:
        try:
            download_from_huggingface(
                model_source=model_name,
                staging_dir=staging_dir,
                timeout_seconds=timeout_seconds,
            )
            result = finalize_download(staging_dir=staging_dir, target_dir=target_dir)
            log(f"Embedding model download complete via Hugging Face: {result}")
            return result
        except Exception:
            safe_rmtree(staging_dir)
            raise
    except Exception:
        safe_rmtree(staging_dir)
        raise


def print_runtime_context(backend: str) -> None:
    """Log runtime context needed to diagnose environment mismatches."""
    env_name = current_conda_env()
    log(f"Active conda environment: {env_name or '<none>'}")
    log(f"Python executable: {sys.executable}")
    log(f"Platform: {sys.platform}")
    log(f"Backend: {backend}")
    log("PyTorch install strategy: bootstrap.sh owns torch installation; init.py never installs torch/torchvision/torchaudio")
    log(f"Preferred pip index-url: {pip_index_url()}")
    extras = pip_extra_index_urls()
    if extras:
        log(f"Preferred pip extra-index-url: {' '.join(extras)}")
    else:
        log("Preferred pip extra-index-url: <none>")
    log(f"nvidia-smi available: {command_exists('nvidia-smi')}")


def main() -> None:
    """Run dependency installation and model preparation."""
    args = parse_args()
    config = load_config_values()
    backend = effective_backend(config)

    ensure_directories()
    print_runtime_context(backend=backend)

    if not args.skip_install:
        install_packages(force=args.force, backend=backend, config=config)

    if not args.skip_model:
        download_model(
            model_source=config["llm.service.model_source"],
            model_source_cn=config["llm.service.model_source_cn"],
            model_cache_dir=config["llm.service.model_cache_dir"],
            timeout_seconds=int(config["llm.service.download_timeout"]),
            force=args.force,
        )
        if config["retrieval.enabled"].strip().lower() == "true":
            download_embedding_model(
                model_name=config["retrieval.model_name"],
                model_cache_dir=config["retrieval.model_cache_dir"],
                timeout_seconds=int(config["llm.service.download_timeout"]),
                force=args.force,
            )

    log("")
    log("Initialization complete.")
    log(f"Backend: {backend}")
    log("Next step: run the platform start launcher.")


if __name__ == "__main__":
    main()
