"""Bootstrap script for the local LangGraph tool-use project."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from app.common.settings import DEFAULT_CONFIG_PATH, get_settings


ROOT_DIR = Path(__file__).resolve().parent


def ensure_directories() -> None:
    """Create expected project directories if missing."""
    directories = [
        ROOT_DIR / "config",
        ROOT_DIR / "scripts",
        ROOT_DIR / "app" / "common",
        ROOT_DIR / "app" / "llm_client",
        ROOT_DIR / "app" / "mcp_server" / "docs",
        ROOT_DIR / "app" / "mcp_server" / "tools",
        ROOT_DIR / "app" / "gateway",
        ROOT_DIR / "app" / "graph",
        ROOT_DIR / "logs",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def create_conda_environment(environment_name: str, python_version: str) -> None:
    """Create a conda environment if conda is available."""
    command = [
        "conda",
        "create",
        "-n",
        environment_name,
        f"python={python_version}",
        "-y",
    ]
    print("Creating conda environment:")
    print(" ".join(command))
    subprocess.run(command, check=False)


def install_packages(environment_name: str) -> None:
    """Install the required Python packages with conda run."""
    packages = [
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "pyyaml",
        "langgraph",
        "langchain-openai",
        "vllm",
    ]
    command = ["conda", "run", "-n", environment_name, "pip", "install", *packages]
    print("Installing packages:")
    print(" ".join(command))
    subprocess.run(command, check=False)


def print_model_commands() -> None:
    """Print suggested model preparation commands."""
    settings = get_settings(str(DEFAULT_CONFIG_PATH))
    model_source = settings.llm.service.model_source
    print("\nSuggested model preparation commands:")
    print(f"  huggingface-cli download {model_source}")
    print(f"  python -m vllm.entrypoints.openai.api_server --model {model_source} --port {settings.llm.service.port}")


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Initialize the local LangGraph tool-use project.")
    parser.add_argument("--skip-conda", action="store_true", help="Skip conda environment creation.")
    parser.add_argument("--skip-install", action="store_true", help="Skip package installation.")
    return parser.parse_args()


def main() -> None:
    """Run project bootstrap steps."""
    args = parse_args()
    settings = get_settings(str(DEFAULT_CONFIG_PATH))
    ensure_directories()

    if not args.skip_conda:
        create_conda_environment(
            environment_name=settings.project.environment_name,
            python_version=settings.project.python_version,
        )
    if not args.skip_install:
        install_packages(settings.project.environment_name)

    print_model_commands()
    print("\nInitialization complete.")
    print(f"Config file: {DEFAULT_CONFIG_PATH}")


if __name__ == "__main__":
    main()
