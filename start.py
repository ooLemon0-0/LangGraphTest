"""Cross-platform runtime launcher for the full project."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description="Start the local LangGraph tool-use project.")
    parser.add_argument("--skip-health-check", action="store_true", help="Do not wait for service health endpoints.")
    return parser.parse_args()


from app.common.settings import get_settings  # noqa: E402


def backend_for_platform(configured_backend: str) -> str:
    """Resolve the backend actually used by this platform."""
    if configured_backend == "vllm" and os.name == "nt":
        return "transformers"
    if configured_backend == "vllm" and sys.platform == "darwin":
        return "transformers"
    return configured_backend


def launch_process(command: list[str], log_path: Path) -> tuple[subprocess.Popen[str], object]:
    """Start one child process and redirect logs."""
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process, handle


def wait_for_health(url: str, timeout_seconds: int = 240) -> None:
    """Poll an HTTP health endpoint until it succeeds."""
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Health check failed for {url}. Last error: {last_error}")


def terminate_processes(processes: list[subprocess.Popen[str]]) -> None:
    """Terminate all child processes."""
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 10
    while time.time() < deadline:
        if all(process.poll() is not None for process in processes):
            return
        time.sleep(0.5)

    for process in processes:
        if process.poll() is None:
            process.kill()


def main() -> None:
    """Launch the LLM service, MCP server, and gateway."""
    args = parse_args()
    settings = get_settings()
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    python_executable = os.environ.get("PYTHON_EXECUTABLE", sys.executable)
    llm_backend = backend_for_platform(settings.llm.service.server_backend.strip().lower())

    llm_command = [
        python_executable,
        "-m",
        "app.llm_server.main",
    ]
    if llm_backend == "vllm":
        llm_command = [
            python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            settings.llm.service.host,
            "--port",
            str(settings.llm.service.port),
            "--model",
            settings.llm.service.model_source,
            "--dtype",
            settings.llm.service.dtype,
            "--tensor-parallel-size",
            str(settings.llm.service.tensor_parallel_size),
            "--download-dir",
            str((ROOT_DIR / settings.llm.service.model_cache_dir).resolve()),
        ]

    mcp_command = [
        python_executable,
        "-m",
        "uvicorn",
        "app.mcp_server.main:app",
        "--host",
        settings.mcp.service.host,
        "--port",
        str(settings.mcp.service.port),
    ]
    gateway_command = [
        python_executable,
        "-m",
        "uvicorn",
        "app.gateway.main:app",
        "--host",
        settings.gateway.service.host,
        "--port",
        str(settings.gateway.service.port),
    ]

    commands = [
        ("llm", llm_command, f"http://{settings.llm.service.host}:{settings.llm.service.port}/health"),
        ("mcp", mcp_command, f"http://{settings.mcp.service.host}:{settings.mcp.service.port}/health"),
        ("gateway", gateway_command, f"http://{settings.gateway.service.host}:{settings.gateway.service.port}/health"),
    ]

    processes: list[subprocess.Popen[str]] = []
    log_handles: list[object] = []
    try:
        print("[start] Preparing services:")
        print(
            f"[start]   llm -> http://{settings.llm.service.host}:{settings.llm.service.port} "
            f"(log: {logs_dir / 'llm.log'})"
        )
        print(
            f"[start]   mcp -> http://{settings.mcp.service.host}:{settings.mcp.service.port} "
            f"(log: {logs_dir / 'mcp.log'})"
        )
        print(
            f"[start]   gateway -> http://{settings.gateway.service.host}:{settings.gateway.service.port} "
            f"(log: {logs_dir / 'gateway.log'})"
        )
        print("[start] Starting services...")

        for name, command, health_url in commands:
            process, log_handle = launch_process(command, logs_dir / f"{name}.log")
            processes.append(process)
            log_handles.append(log_handle)
            print(f"[start] {name} started with PID {process.pid}")
            if not args.skip_health_check:
                wait_for_health(health_url)
                print(f"[start] {name} health check passed: {health_url}")

        print()
        print("[start] All services are running.")
        print(f"[start] Gateway: http://{settings.gateway.service.host}:{settings.gateway.service.port}")
        print(f"[start] Logs: {logs_dir}")
        print("[start] Press Ctrl+C to stop all services.")

        while True:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Child process exited early with code {process.returncode}.")
            time.sleep(2)
    except KeyboardInterrupt:
        print()
        print("[start] Stopping services...")
    finally:
        terminate_processes(processes)
        for handle in log_handles:
            handle.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
