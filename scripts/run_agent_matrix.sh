#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLASSIC_SCENARIOS="cartpole_balance mountain_car acrobot_swingup pendulum_swingup"
BOX2D_SCENARIOS="lunar_lander lunar_lander_continuous bipedal_walker car_racing"
MINIGRID_SCENARIOS="minigrid_doorkey_16x16 minigrid_keycorridor_s6r3 minigrid_obstructedmaze_2dlhb minigrid_lavacrossing_s11n5"
MUJOCO_SCENARIOS="reacher inverted_pendulum hopper half_cheetah"
IMPLEMENTED_SCENARIOS="$CLASSIC_SCENARIOS $BOX2D_SCENARIOS $MINIGRID_SCENARIOS $MUJOCO_SCENARIOS"

SCENARIO_SET="${SCENARIO_SET:-implemented}"
case "$SCENARIO_SET" in
  classic) DEFAULT_SCENARIOS="$CLASSIC_SCENARIOS" ;;
  box2d) DEFAULT_SCENARIOS="$BOX2D_SCENARIOS" ;;
  minigrid) DEFAULT_SCENARIOS="$MINIGRID_SCENARIOS" ;;
  mujoco) DEFAULT_SCENARIOS="$MUJOCO_SCENARIOS" ;;
  implemented | core16) DEFAULT_SCENARIOS="$IMPLEMENTED_SCENARIOS" ;;
  *)
    echo "Unknown SCENARIO_SET=${SCENARIO_SET}. Expected classic, box2d, minigrid, mujoco, implemented, or core16." >&2
    exit 2
    ;;
esac

SCENARIOS="${SCENARIOS:-$DEFAULT_SCENARIOS}"
AGENT_MODELS="${AGENT_MODELS:-gpt-5.4-mini}"
MODEL_PREFIX="${MODEL_PREFIX:-codex}"
EPOCHS="${EPOCHS:-8}"
TRAIN_EPISODES="${TRAIN_EPISODES:-32}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
PRESET="${PRESET:-smoke}"
JOBS="${JOBS:-1}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y%m%d-%H%M%S)}"
PROBE_EPISODES="${PROBE_EPISODES:-16}"
AGENT_COMMAND_TEMPLATE="${AGENT_COMMAND_TEMPLATE:-}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_SHARED_MODEL_NAME="${ALLOW_SHARED_MODEL_NAME:-0}"

usage() {
  cat <<'EOF'
Run an HLBench model/environment matrix with bounded parallelism.

Environment variables:
  SCENARIO_SET=implemented       One of classic, box2d, minigrid, mujoco, implemented/core16.
  SCENARIOS="a b c"              Override the scenario list.
  AGENT_MODELS="gpt-5.4-mini"    Agent model ids to evaluate.
  MODEL_PREFIX=codex             Prefix for generated model-name directories.
  MODEL_NAME=<name>              Optional fixed model-name; only safe with one AGENT_MODEL.
  EPOCHS=8                       Harness epochs per model/env cell.
  TRAIN_EPISODES=32              Train episodes sampled per epoch.
  TIMEOUT_SECONDS=900            Agent command timeout per epoch.
  JOBS=1                         Number of model/env cells to run in parallel.
  RUN_GROUP=<timestamp>          Shared run id prefix and log group.
  DRY_RUN=1                      Print planned cells without running them.
  AGENT_COMMAND_TEMPLATE=...     Optional command template with {AGENT_MODEL}, {SCENARIO}, {PROMPT}.

Examples:
  JOBS=4 EPOCHS=8 TRAIN_EPISODES=32 ./scripts/run_agent_matrix.sh
  SCENARIO_SET=minigrid AGENT_MODELS="gpt-5.4-mini gpt-5.4" JOBS=2 ./scripts/run_agent_matrix.sh
  DRY_RUN=1 SCENARIOS="mountain_car car_racing" ./scripts/run_agent_matrix.sh
  tail -f runs/_matrix_logs/<run_group>/status.tsv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

sanitize() {
  printf "%s" "$1" | tr -c "A-Za-z0-9_.-" "-"
}

word_count() {
  local count=0
  local item
  for item in $1; do
    count=$((count + 1))
  done
  printf "%s" "$count"
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
}

model_name_for() {
  local agent_model="$1"
  if [[ -n "${MODEL_NAME:-}" ]]; then
    printf "%s" "$MODEL_NAME"
  else
    sanitize "${MODEL_PREFIX}-${agent_model}"
  fi
}

agent_prompt() {
  local scenario="$1"
  printf "%s" "Optimize this HLBench ${scenario} policy. Read AGENTS.md, task.md, task_contract.json, and feedback if present. Edit only system/policy.py. Preserve the required interface exactly: class Policy, reset(self, task_config), and act(self, observation, context). You may run train-only rollouts with: python -m hlbench.rollout.cli --workspace . --split train --episodes ${PROBE_EPISODES} --output-dir experiments/<name>. You may use aggregate validation summaries exposed under feedback/history, but must not inspect validation seeds, validation replays, validation per-episode records, validation failure details, heldout metrics, or heldout data. Do not modify task.md, task_contract.json, AGENTS.md, or feedback. Avoid broad repository searches and unbounded parameter sweeps. Finish with a valid policy."
}

agent_command_for() {
  local agent_model="$1"
  local scenario="$2"
  local prompt
  prompt="$(agent_prompt "$scenario")"
  if [[ -n "$AGENT_COMMAND_TEMPLATE" ]]; then
    local command="$AGENT_COMMAND_TEMPLATE"
    command="${command//\{AGENT_MODEL\}/$agent_model}"
    command="${command//\{SCENARIO\}/$scenario}"
    command="${command//\{PROMPT\}/$prompt}"
    printf "%s" "$command"
    return
  fi
  printf "%s" "codex exec -m ${agent_model} --skip-git-repo-check --sandbox workspace-write --ephemeral '${prompt}'"
}

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

append_status() {
  local status="$1"
  local agent_model="$2"
  local model_name="$3"
  local scenario="$4"
  local run_id="$5"
  local duration_seconds="$6"
  local exit_code="$7"
  local log_file="$8"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(timestamp_utc)" "$status" "$agent_model" "$model_name" "$scenario" "$run_id" \
    "$duration_seconds" "$exit_code" "$log_file" >>"$STATUS_FILE"
}

run_one() {
  local agent_model="$1"
  local scenario="$2"
  local model_name
  model_name="$(model_name_for "$agent_model")"
  local run_id="${RUN_ID:-${RUN_GROUP}-e${EPOCHS}-train${TRAIN_EPISODES}}"
  local log_dir="$ROOT_DIR/runs/_matrix_logs/$RUN_GROUP"
  local log_file="$log_dir/${model_name}__${scenario}.log"
  local agent_command
  agent_command="$(agent_command_for "$agent_model" "$scenario")"
  mkdir -p "$log_dir"

  echo "[start] model=${model_name} agent_model=${agent_model} scenario=${scenario} log=${log_file}"
  local started
  started="$(date +%s)"
  append_status "start" "$agent_model" "$model_name" "$scenario" "$run_id" "0" "" "$log_file"
  if PYTHONPATH=src python -m hlbench run \
    --scenario "$scenario" \
    --model-name "$model_name" \
    --run-id "$run_id" \
    --epochs "$EPOCHS" \
    --preset "$PRESET" \
    --train-episodes "$TRAIN_EPISODES" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --agent-backend command \
    --agent-command "$agent_command" \
    >"$log_file" 2>&1; then
    local duration
    duration="$(($(date +%s) - started))"
    append_status "ok" "$agent_model" "$model_name" "$scenario" "$run_id" "$duration" "0" "$log_file"
    echo "[ok] model=${model_name} scenario=${scenario} duration=${duration}s"
    return 0
  else
    local status=$?
    local duration
    duration="$(($(date +%s) - started))"
    append_status "fail" "$agent_model" "$model_name" "$scenario" "$run_id" "$duration" "$status" "$log_file"
    echo "[fail] model=${model_name} scenario=${scenario} status=${status} duration=${duration}s log=${log_file}"
    tail -n 40 "$log_file" || true
    return "$status"
  fi
}

validate_positive_int "EPOCHS" "$EPOCHS"
validate_positive_int "TRAIN_EPISODES" "$TRAIN_EPISODES"
validate_positive_int "TIMEOUT_SECONDS" "$TIMEOUT_SECONDS"
validate_positive_int "JOBS" "$JOBS"
validate_positive_int "PROBE_EPISODES" "$PROBE_EPISODES"
if [[ "$(word_count "$SCENARIOS")" == "0" ]]; then
  echo "SCENARIOS must not be empty." >&2
  exit 2
fi
if [[ "$(word_count "$AGENT_MODELS")" == "0" ]]; then
  echo "AGENT_MODELS must not be empty." >&2
  exit 2
fi
if [[ -n "${MODEL_NAME:-}" && "$(word_count "$AGENT_MODELS")" != "1" && "$ALLOW_SHARED_MODEL_NAME" != "1" ]]; then
  echo "MODEL_NAME is fixed but AGENT_MODELS contains multiple models. Unset MODEL_NAME or set ALLOW_SHARED_MODEL_NAME=1." >&2
  exit 2
fi

if [[ "${HLBENCH_MATRIX_WORKER:-0}" == "1" ]]; then
  run_one "$AGENT_MODEL" "$SCENARIO"
  exit $?
fi

export AGENT_COMMAND_TEMPLATE ALLOW_SHARED_MODEL_NAME DRY_RUN EPOCHS HLBENCH_MATRIX_WORKER JOBS MODEL_PREFIX PRESET PROBE_EPISODES
export RUN_GROUP SCENARIO_SET TIMEOUT_SECONDS TRAIN_EPISODES
if [[ -n "${MODEL_NAME:-}" ]]; then
  export MODEL_NAME
fi
if [[ -n "${RUN_ID:-}" ]]; then
  export RUN_ID
fi

LOG_DIR="$ROOT_DIR/runs/_matrix_logs/$RUN_GROUP"
TASKS_FILE="$LOG_DIR/tasks.tsv"
STATUS_FILE="$LOG_DIR/status.tsv"
SUMMARY_FILE="$LOG_DIR/summary.txt"
export STATUS_FILE

echo "HLBench matrix run"
echo "  run_group=${RUN_GROUP}"
echo "  scenarios=${SCENARIOS}"
echo "  agent_models=${AGENT_MODELS}"
echo "  jobs=${JOBS}"
echo "  epochs=${EPOCHS}"
echo "  train_episodes=${TRAIN_EPISODES}"
echo "  timeout_seconds=${TIMEOUT_SECONDS}"
echo "  logs=runs/_matrix_logs/${RUN_GROUP}"

mkdir -p "$LOG_DIR"
printf "agent_model\tmodel_name\tscenario\trun_id\tlog_file\n" >"$TASKS_FILE"
printf "timestamp\tstatus\tagent_model\tmodel_name\tscenario\trun_id\tduration_seconds\texit_code\tlog_file\n" >"$STATUS_FILE"

while read -r agent_model scenario; do
  model_name="$(model_name_for "$agent_model")"
  run_id="${RUN_ID:-${RUN_GROUP}-e${EPOCHS}-train${TRAIN_EPISODES}}"
  log_file="runs/_matrix_logs/${RUN_GROUP}/${model_name}__${scenario}.log"
  printf "%s\t%s\t%s\t%s\t%s\n" "$agent_model" "$model_name" "$scenario" "$run_id" "$log_file" >>"$TASKS_FILE"
done < <(
  for agent_model in $AGENT_MODELS; do
    for scenario in $SCENARIOS; do
      printf "%s %s\n" "$agent_model" "$scenario"
    done
  done
)

echo "  task_manifest=runs/_matrix_logs/${RUN_GROUP}/tasks.tsv"
echo "  status_log=runs/_matrix_logs/${RUN_GROUP}/status.tsv"

if [[ "$DRY_RUN" == "1" ]]; then
  cat "$TASKS_FILE"
  exit 0
fi

set +e
tail -n +2 "$TASKS_FILE" | cut -f1,3 | xargs -n 2 -P "$JOBS" bash -c 'AGENT_MODEL="$1" SCENARIO="$2" HLBENCH_MATRIX_WORKER=1 "$0"' "$0"
matrix_status=$?
set -e

awk -F '\t' '
  NR > 1 && $2 == "start" { started += 1 }
  NR > 1 && $2 == "ok" { ok += 1 }
  NR > 1 && $2 == "fail" { fail += 1 }
  END {
    printf("started\t%d\nok\t%d\nfail\t%d\n", started + 0, ok + 0, fail + 0)
  }
' "$STATUS_FILE" >"$SUMMARY_FILE"
cat "$SUMMARY_FILE"
exit "$matrix_status"
