"""Initialize project dependencies and model assets inside the active conda environment."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"
DEFAULTS = {
    "llm.service.server_backend": "transformers",
    "llm.service.model_source": "Qwen/Qwen3-1.7B",
    "llm.service.model_cache_dir": "models",
}


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

    install_command = [sys.executable, "-m", "pip", "install"]
    if force:
        install_command.append("--upgrade")
    install_command.extend(["-r", "requirements.txt"])
    run_command(install_command, "Installing project dependencies")

    if backend == "vllm":
        vllm_command = [sys.executable, "-m", "pip", "install"]
        if force:
            vllm_command.append("--upgrade")
        vllm_command.append("vllm")
        run_command(vllm_command, "Installing optional vLLM backend")


def download_model(model_source: str, model_cache_dir: str) -> None:
    """Download the configured Hugging Face model to the local cache."""
    cache_dir = (ROOT_DIR / model_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    download_script = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download(repo_id={model_source!r}, cache_dir={str(cache_dir)!r}, resume_download=True)"
    )
    run_command([sys.executable, "-c", download_script], f"Downloading model {model_source}")


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
            model_cache_dir=config["llm.service.model_cache_dir"],
        )

    print()
    print("[init] Initialization complete.")
    print(f"[init] Backend: {backend}")
    print("[init] Next step: run the platform start launcher.")


if __name__ == "__main__":
    main()
