#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PREFIX="${MODEL_PREFIX:-kimi}"
export AGENT_MODELS="${AGENT_MODELS:-kimi-code/kimi-for-coding}"
if [[ -z "${AGENT_COMMAND_TEMPLATE:-}" ]]; then
  AGENT_COMMAND_TEMPLATE="kimi --print --model {AGENT_MODEL} --final-message-only --prompt '{PROMPT}'"
  export AGENT_COMMAND_TEMPLATE
fi

exec "$SCRIPT_DIR/run_agent_matrix.sh" "$@"
