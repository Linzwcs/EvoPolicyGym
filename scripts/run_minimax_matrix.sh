#!/usr/bin/env bash
set -euo pipefail

# Drive the agent matrix with MiniMax M2.7 through the Claude Code CLI.
# Prerequisite: ~/.claude/settings.json must point ANTHROPIC_BASE_URL and
# ANTHROPIC_AUTH_TOKEN at MiniMax. Run scripts/configure_claude_minimax.sh
# once before using this. Also ensure ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL
# are NOT set in the current shell (they would override settings.json).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PREFIX="${MODEL_PREFIX:-minimax}"
export AGENT_MODELS="${AGENT_MODELS:-MiniMax-M2.7}"
if [[ -z "${AGENT_COMMAND_TEMPLATE:-}" ]]; then
  AGENT_COMMAND_TEMPLATE="claude -p --model {AGENT_MODEL} --permission-mode bypassPermissions --tools default --no-session-persistence --output-format text '{PROMPT}'"
  export AGENT_COMMAND_TEMPLATE
fi

exec "$SCRIPT_DIR/run_agent_matrix.sh" "$@"
