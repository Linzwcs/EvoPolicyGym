#!/usr/bin/env bash
# scripts/run_v1_paper_matrix_codex.sh — v1 paper 16-env matrix runner
# (server-friendly, OpenAI Codex backend).
#
# Same env roster, parallelism shape, and run-dir layout as
# scripts/run_v1_paper_matrix.sh — only the agent driver differs.
# Drives ``hlbench agent --backend codex`` instead of the Claude Code
# default. Both scripts can run side-by-side under different
# `MODEL_SLUG`s without clobbering each other's run dirs.
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
#   # default: gpt-5-codex × budget=64 × max-turns=8 × parallel=4
#   bash scripts/run_v1_paper_matrix_codex.sh
#
#   # alternative codex model (must be a model your codex-cli is
#   # configured to call — see `codex --help`)
#   bash scripts/run_v1_paper_matrix_codex.sh --model o3
#
#   # heavier eval (paper-grade)
#   bash scripts/run_v1_paper_matrix_codex.sh --budget 256 --max-turns 24
#
#   # different parallelism
#   bash scripts/run_v1_paper_matrix_codex.sh --max-parallel 8
#
#   # nohup background (survives SSH disconnect)
#   nohup bash scripts/run_v1_paper_matrix_codex.sh > nohup-codex.out 2>&1 &
#
# Env-var fallbacks (let cron / CI set without flags):
#   MODEL=o3 BUDGET=256 MAX_TURNS=24 \
#       bash scripts/run_v1_paper_matrix_codex.sh
#
# Auth: codex needs `~/.codex/auth` (run `codex login` once) or
# OPENAI_API_KEY in the environment. The harness inherits whatever
# the local `codex` CLI is configured with.

set -euo pipefail

# ===================================================================
# v1 paper Table 1 — 16 envs across 4 Gymnasium categories.
# Identical roster to run_v1_paper_matrix.sh so cross-backend
# numbers are directly comparable in paper/table.md.
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
#
# Differences vs run_v1_paper_matrix.sh:
#   MODEL          sonnet           → gpt-5-codex
#   MODEL_SLUG     claude-code-auto → codex-auto
#   EXP_ID prefix  v1paper-<model>  → v1paper-codex-<model>
#   TURN_TIMEOUT default the same 900s (codex first-call latency on
#                 macOS commonly busts the 600s claude default; the
#                 hlbench agent CLI already auto-bumps to 900 for
#                 codex but we set it explicitly here too so the
#                 matrix runner's per-process default doesn't drift).
# ===================================================================
MODEL="${MODEL:-gpt-5-codex}"
MODEL_SLUG="${MODEL_SLUG:-codex-auto}"
BUDGET="${BUDGET:-64}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TURN_TIMEOUT="${TURN_TIMEOUT:-900}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
EXP_ID="${EXP_ID:-v1paper-codex-${MODEL}-$(date +%Y%m%d-%H%M%S)}"
CODEX_BINARY="${CODEX_BINARY:-codex}"

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

if ! command -v "${CODEX_BINARY}" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: '${CODEX_BINARY}' (OpenAI Codex CLI) binary not on PATH.

Install:
    npm i -g @openai/codex
    # (or follow the instructions at https://github.com/openai/codex)

Verify:
    ${CODEX_BINARY} --version

Auth (one of):
    ${CODEX_BINARY} login                  # interactive, writes ~/.codex/auth
    export OPENAI_API_KEY=sk-...           # API-key fallback

If your codex binary lives elsewhere, point the script at it:
    CODEX_BINARY=/path/to/codex bash scripts/run_v1_paper_matrix_codex.sh
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
        --codex-binary)  shift; CODEX_BINARY="$1" ;;
        -h|--help)
            sed -n '/^# Usage/,/^# Auth:/p' "$0" | sed 's/^#\s\?//'
            exit 0
            ;;
        *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# Re-derive EXP_ID if MODEL was overridden after the default was set
# (and the user didn't explicitly set an exp_id).
case "${EXP_ID}" in
    v1paper-codex-gpt-5-codex-*)
        if [ "${MODEL}" != "gpt-5-codex" ]; then
            EXP_ID="v1paper-codex-${MODEL}-$(date +%Y%m%d-%H%M%S)"
        fi
        ;;
esac

# ===================================================================
# Announce.
# ===================================================================
N_ENVS=${#V1_PAPER_ENVS[@]}

cat <<EOF
============================================================
hlbench-pro — v1 paper Table 1 matrix runner (Codex backend)
============================================================
  backend:       codex (OpenAI Codex CLI)
  codex binary:  ${CODEX_BINARY} ($(${CODEX_BINARY} --version 2>/dev/null | head -1))
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
  60-120 min for default budget=64/max-turns=8 at parallel=4. Codex
  first-call latency on a fresh box can add a minute or two per env.

Cost: Codex 0.133's --json doesn't surface per-turn token counts,
so this script can't estimate cost mid-flight. Settle up against
your OpenAI billing dashboard. Reference points (gpt-5-codex,
2026-05 pricing): a budget=64 / max-turns=8 run is roughly
\$0.50-\$2.00 per env, scaling roughly linearly with budget.

Side note: codex persists session rollouts at
~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl outside the run dir.
Won't affect correctness; just disk to be aware of on long matrices.

EOF

# ===================================================================
# Launch. exec so signals (Ctrl-C, SIGTERM from nohup-bg) reach
# python and propagate to its hlbench-agent children.
# ===================================================================
exec "${VENV_PY}" "${REPO_ROOT}/scripts/run_matrix.py" \
    --envs "${V1_PAPER_ENVS[@]}" \
    --backend codex \
    --codex-binary "${CODEX_BINARY}" \
    --budget "${BUDGET}" \
    --max-turns "${MAX_TURNS}" \
    --model "${MODEL}" \
    --model-slug "${MODEL_SLUG}" \
    --turn-timeout "${TURN_TIMEOUT}" \
    --runs-root "${RUNS_ROOT}" \
    --exp-id "${EXP_ID}" \
    --max-parallel "${MAX_PARALLEL}"
