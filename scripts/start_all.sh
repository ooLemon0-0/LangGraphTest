#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

cleanup() {
  if [[ -n "${LLM_PID:-}" ]]; then kill "$LLM_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${MCP_PID:-}" ]]; then kill "$MCP_PID" >/dev/null 2>&1 || true; fi
  if [[ -n "${GATEWAY_PID:-}" ]]; then kill "$GATEWAY_PID" >/dev/null 2>&1 || true; fi
}

trap cleanup EXIT INT TERM

"${ROOT_DIR}/scripts/start_llm.sh" >"${LOG_DIR}/llm.log" 2>&1 &
LLM_PID=$!

"${ROOT_DIR}/scripts/start_mcp.sh" >"${LOG_DIR}/mcp.log" 2>&1 &
MCP_PID=$!

"${ROOT_DIR}/scripts/start_gateway.sh" >"${LOG_DIR}/gateway.log" 2>&1 &
GATEWAY_PID=$!

echo "LLM PID: ${LLM_PID}"
echo "MCP PID: ${MCP_PID}"
echo "Gateway PID: ${GATEWAY_PID}"
echo "Logs: ${LOG_DIR}"

wait
