#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PREFIX="${MODEL_PREFIX:-codex}"
export AGENT_MODELS="${AGENT_MODELS:-gpt-5.4-mini}"
export AGENT_COMMAND_TEMPLATE="${AGENT_COMMAND_TEMPLATE:-codex exec -m {AGENT_MODEL} --skip-git-repo-check --sandbox workspace-write --ephemeral '{PROMPT}'}"

exec "$SCRIPT_DIR/run_agent_matrix.sh" "$@"
