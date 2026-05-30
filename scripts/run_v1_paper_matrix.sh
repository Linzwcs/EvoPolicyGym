#!/usr/bin/env bash
# scripts/run_v1_paper_matrix.sh — v1 paper 16-env matrix runner (server-friendly).
#
# Hardcodes the v1 scored suite from paper/table.md (16 envs across
# 4 Gymnasium categories) and wraps scripts/run_matrix.py with
# server-side defaults: parallel=4, max_turns=8, budget=64, sonnet.
#
# Designed for nohup'd background execution on a remote server:
#   - timestamped exp_id (multiple runs don't clobber each other)
#   - all logs under runs/_matrix_logs/<exp_id>__*.log
#   - per-env run.json under runs/<model-slug>/<env>/<exp_id>/run.json
#   - summary table printed to stdout at end
#   - exit code 0 iff every env's outcome.status == "completed"
#
# Usage (one-liner from repo root):
#
#   # default: sonnet × budget=64 × max-turns=8 × parallel=4 (~$25, ~90 min)
#   bash scripts/run_v1_paper_matrix.sh
#
#   # opus
#   bash scripts/run_v1_paper_matrix.sh --model opus
#
#   # heavier eval (paper-grade)
#   bash scripts/run_v1_paper_matrix.sh --budget 256 --max-turns 24
#
#   # different parallelism
#   bash scripts/run_v1_paper_matrix.sh --max-parallel 8
#
#   # nohup background (survives SSH disconnect)
#   nohup bash scripts/run_v1_paper_matrix.sh --model opus > nohup-opus.out 2>&1 &
#
# Env-var fallbacks (let cron / CI set without flags):
#   MODEL=opus BUDGET=256 MAX_TURNS=24 bash scripts/run_v1_paper_matrix.sh

set -euo pipefail

# ===================================================================
# v1 paper Table 1 — 16 envs across 4 Gymnasium categories.
# Order matters only for log readability; run_matrix.py shuffles
# slot order based on completion time.
# ===================================================================
V1_PAPER_ENVS=(
    # Classic Control (4)
    cartpole_balance
    pendulum
    acrobot
    mountain_car_continuous
    # Box2D (4) — 2 hardcore variants + 2 CarRacing (lite + full)
    lunar_hardcore
    bipedal_hardcore
    car_racing
    car_racing_pixel
    # MuJoCo locomotion (4)
    half_cheetah
    hopper
    walker2d
    ant
    # MiniGrid POMDP navigation (4)
    minigrid_doorkey
    minigrid_keycorridor
    minigrid_lavacrossing
    minigrid_obstructedmaze
)

# Repo root from script location (works from any cwd).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ===================================================================
# Defaults (overridable by env var or CLI flag).
# ===================================================================
MODEL="${MODEL:-sonnet}"
MODEL_SLUG="${MODEL_SLUG:-claude-code-auto}"
BUDGET="${BUDGET:-64}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TURN_TIMEOUT="${TURN_TIMEOUT:-900}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
EXP_ID="${EXP_ID:-v1paper-${MODEL}-$(date +%Y%m%d-%H%M%S)}"

# ===================================================================
# Preflight: tools that must exist on PATH / in venv.
# ===================================================================
VENV_PY="${REPO_ROOT}/.venv/bin/python"
HLBENCH_BIN="${REPO_ROOT}/.venv/bin/hlbench"

preflight_fail=0

if [ ! -x "${VENV_PY}" ]; then
    cat >&2 <<EOF
ERROR: no Python venv at ${VENV_PY}

Bootstrap (from repo root):
    uv venv --python 3.12 .venv
    uv pip install --python ${VENV_PY} -e .
    uv pip install --python ${VENV_PY} mujoco minigrid pytest ruff mypy

EOF
    preflight_fail=1
fi

if [ ! -x "${HLBENCH_BIN}" ]; then
    cat >&2 <<EOF
ERROR: hlbench CLI not installed in venv at ${HLBENCH_BIN}

Fix:
    uv pip install --python ${VENV_PY} -e .
EOF
    preflight_fail=1
fi

if ! command -v claude >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: 'claude' (Claude Code) binary not on PATH.

Install: https://claude.com/claude-code/download
For the OpenAI Codex backend instead, install 'codex'
('npm i -g @openai/codex') and pass --backend codex through
run_matrix.py (it now supports the flag).
EOF
    preflight_fail=1
fi

if [ "${preflight_fail}" = "1" ]; then
    exit 2
fi

# ===================================================================
# CLI flag parsing (overrides env-var defaults).
# ===================================================================
while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)         shift; MODEL="$1" ;;
        --model-slug)    shift; MODEL_SLUG="$1" ;;
        --budget)        shift; BUDGET="$1" ;;
        --max-turns)     shift; MAX_TURNS="$1" ;;
        --max-parallel)  shift; MAX_PARALLEL="$1" ;;
        --turn-timeout)  shift; TURN_TIMEOUT="$1" ;;
        --runs-root)     shift; RUNS_ROOT="$1" ;;
        --exp-id)        shift; EXP_ID="$1" ;;
        -h|--help)
            sed -n '/^# Usage/,/^# Env-var/p' "$0" | sed 's/^#\s\?//'
            exit 0
            ;;
        *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# Re-derive EXP_ID if MODEL was overridden after the default was set
# (and the user didn't explicitly set an exp_id).
case "${EXP_ID}" in
    v1paper-sonnet-*)
        if [ "${MODEL}" != "sonnet" ]; then
            EXP_ID="v1paper-${MODEL}-$(date +%Y%m%d-%H%M%S)"
        fi
        ;;
esac

# ===================================================================
# Announce.
# ===================================================================
N_ENVS=${#V1_PAPER_ENVS[@]}

cat <<EOF
============================================================
hlbench-pro — v1 paper Table 1 matrix runner
============================================================
  envs:          ${N_ENVS} (v1 paper scored suite)
                 ${V1_PAPER_ENVS[@]}
  model:         ${MODEL}
  model-slug:    ${MODEL_SLUG}
  budget/env:    ${BUDGET} episodes
  max-turns/env: ${MAX_TURNS}
  turn-timeout:  ${TURN_TIMEOUT}s
  max-parallel:  ${MAX_PARALLEL}
  runs-root:     ${RUNS_ROOT}
  exp-id:        ${EXP_ID}
============================================================

Per-env logs:       ${RUNS_ROOT}/_matrix_logs/${EXP_ID}__<env>.log
Per-env run.json:   ${RUNS_ROOT}/${MODEL_SLUG}/<env>/${EXP_ID}/run.json
Summary printed below at completion.

Wall estimate: ~${MAX_PARALLEL}-cycle of (avg env wall) — typically
  60-120 min for default sonnet/budget=64/max-turns=8 at parallel=4.

Cost estimate (Claude API):
  sonnet × budget=64 × max-turns=8 × 16 envs ≈ \$25-50
  opus   × same                              ≈ \$80-150
  Add ~2x for budget=128, ~4x for budget=256.

EOF

# ===================================================================
# Launch. exec so signals (Ctrl-C, SIGTERM from nohup-bg) reach
# python and propagate to its hlbench-agent children.
# ===================================================================
exec "${VENV_PY}" "${REPO_ROOT}/scripts/run_matrix.py" \
    --envs "${V1_PAPER_ENVS[@]}" \
    --budget "${BUDGET}" \
    --max-turns "${MAX_TURNS}" \
    --model "${MODEL}" \
    --model-slug "${MODEL_SLUG}" \
    --turn-timeout "${TURN_TIMEOUT}" \
    --runs-root "${RUNS_ROOT}" \
    --exp-id "${EXP_ID}" \
    --max-parallel "${MAX_PARALLEL}"
