#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCENARIO="${SCENARIO:-mountain_car}"
MODEL_NAME="${MODEL_NAME:-codex-mini}"
AGENT_MODEL="${AGENT_MODEL:-gpt-5.4-mini}"
EPOCHS="${EPOCHS:-2}"
TRAIN_EPISODES="${TRAIN_EPISODES:-32}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
RUN_ID="${RUN_ID:-${MODEL_NAME}-${SCENARIO}-${EPOCHS}epoch-train${TRAIN_EPISODES}-$(date +%Y%m%d-%H%M%S)}"

PROMPT="Optimize this HLBench ${SCENARIO} policy. Read AGENTS.md, task.md, task_contract.json, and feedback if present. Edit only system/policy.py. Preserve the required interface exactly: class Policy, reset(self, task_config), and act(self, observation, context). You may run train-only rollouts with: python -m hlbench.rollout.cli --workspace . --split train --episodes 16 --output-dir experiments/<name>. You may use aggregate validation summaries exposed under feedback/history, but must not inspect validation seeds, validation replays, validation per-episode records, validation failure details, heldout metrics, or heldout data. Do not modify task.md, task_contract.json, AGENTS.md, or feedback. Avoid broad repository searches and unbounded parameter sweeps. Finish with a valid policy."

PYTHONPATH=src python -m hlbench run \
  --scenario "$SCENARIO" \
  --model-name "$MODEL_NAME" \
  --run-id "$RUN_ID" \
  --epochs "$EPOCHS" \
  --preset smoke \
  --train-episodes "$TRAIN_EPISODES" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --agent-backend command \
  --agent-command "codex exec -m ${AGENT_MODEL} --skip-git-repo-check --sandbox workspace-write --ephemeral '${PROMPT}'"
