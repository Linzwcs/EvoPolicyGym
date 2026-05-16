#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PREFIX="${MODEL_PREFIX:-claude}"
export AGENT_MODELS="${AGENT_MODELS:-sonnet}"
if [[ -z "${AGENT_COMMAND_TEMPLATE:-}" ]]; then
  AGENT_COMMAND_TEMPLATE="claude -p --model {AGENT_MODEL} --permission-mode bypassPermissions --tools default --no-session-persistence --output-format text '{PROMPT}'"
  export AGENT_COMMAND_TEMPLATE
fi

exec "$SCRIPT_DIR/run_agent_matrix.sh" "$@"
