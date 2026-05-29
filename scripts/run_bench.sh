#!/usr/bin/env bash
# scripts/run_bench.sh — one-click hlbench eval matrix runner
#
# Thin wrapper around scripts/run_matrix.py that handles:
#   - preflight (.venv exists, hlbench CLI installed, claude on PATH)
#   - env-var defaults so a server crontab can just run with no args
#   - flag overrides for ad-hoc invocation
#
# Designed for server-side one-click execution. All knobs have
# sensible defaults; just `bash scripts/run_bench.sh` starts a run.
#
# Exit code: 0 iff every env's run.json:outcome.status == "completed".

set -euo pipefail

# Resolve repo root from script location (works from any cwd).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---- defaults (override via env var or CLI flag) ------------------------

# Env list: space-separated. Empty string ⇒ run_matrix.py auto-discovers
# all registered envs.
ENVS="${ENVS:-pendulum lunar_lander_continuous}"
BUDGET="${BUDGET:-256}"
MAX_TURNS="${MAX_TURNS:-24}"
MODEL="${MODEL:-sonnet}"
TURN_TIMEOUT="${TURN_TIMEOUT:-900}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
EXP_ID="${EXP_ID:-bench-$(date +%Y%m%d-%H%M%S)}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"   # 0 = all concurrent

# ---- preflight ----------------------------------------------------------

VENV_PY="${REPO_ROOT}/.venv/bin/python"
HLBENCH_BIN="${REPO_ROOT}/.venv/bin/hlbench"

if [ ! -x "${VENV_PY}" ]; then
    cat >&2 <<EOF
ERROR: no Python venv at ${VENV_PY}

Bootstrap:
    uv venv --python 3.12 .venv
    uv pip install --python ${VENV_PY} -e .
EOF
    exit 2
fi

if [ ! -x "${HLBENCH_BIN}" ]; then
    cat >&2 <<EOF
ERROR: 'hlbench' CLI not installed in venv at ${HLBENCH_BIN}

Fix:
    uv pip install --python ${VENV_PY} -e .
EOF
    exit 2
fi

if ! command -v claude >/dev/null 2>&1; then
    echo "WARNING: 'claude' binary not on PATH — hlbench agent will fail on first turn." >&2
    echo "         Install Claude Code or set --no-require-claude (via run_matrix.py)." >&2
fi

# ---- arg parsing (overrides env-var defaults) ---------------------------

while [ "$#" -gt 0 ]; do
    case "$1" in
        --envs)          shift; ENVS="$1" ;;
        --budget)        shift; BUDGET="$1" ;;
        --max-turns)     shift; MAX_TURNS="$1" ;;
        --model)         shift; MODEL="$1" ;;
        --turn-timeout)  shift; TURN_TIMEOUT="$1" ;;
        --runs-root)     shift; RUNS_ROOT="$1" ;;
        --exp-id)        shift; EXP_ID="$1" ;;
        --max-parallel)  shift; MAX_PARALLEL="$1" ;;
        -h|--help)
            cat <<EOF
Usage: $(basename "$0") [OPTIONS]

One-click eval matrix runner. Spawns N parallel \`hlbench agent\`
processes across the given env list, captures each's log, then prints
a summary table of final_score / cost / wall time per env.

All flags are optional (with env-var fallback in parens):

  --envs  "<id1 id2 ...>"  env ids to run, space-separated
                           (env: ENVS; default: "pendulum lunar_lander_continuous")
  --budget N               episode_budget per env
                           (env: BUDGET; default: 256)
  --max-turns N            agent max_turns per env
                           (env: MAX_TURNS; default: 24)
  --model NAME             claude --model alias or full id
                           (env: MODEL; default: sonnet)
  --turn-timeout SECONDS   seconds per agent turn
                           (env: TURN_TIMEOUT; default: 900)
  --runs-root PATH         where runs/ tree lives
                           (env: RUNS_ROOT; default: ./runs)
  --exp-id ID              shared exp-id across all envs
                           (env: EXP_ID; default: bench-<timestamp>)
  --max-parallel N         max concurrent envs (0 = all)
                           (env: MAX_PARALLEL; default: 0)

  -h, --help               show this help and exit

Examples:
  $(basename "$0")
      # all defaults: 2 envs, budget=256, sonnet, ~14 min, ~\$9

  $(basename "$0") --envs "pendulum" --budget 32 --max-turns 4
      # cheap single-env probe (~5 min, ~\$1)

  ENVS="pendulum acrobot mountain_car_continuous" BUDGET=128 $(basename "$0")
      # 3 envs via env vars

  $(basename "$0") --model haiku --budget 64
      # cheap model on smaller budget

Outputs:
  runs/<model-slug>/<env>/<exp-id>/run.json       — per-env headline
  runs/<model-slug>/<env>/<exp-id>/logs/...       — per-env detailed logs
  runs/_matrix_logs/<exp-id>__<env>.log           — per-env stdout
  (summary table printed to stdout at end)

For server-side detached runs:
  nohup bash $(basename "$0") > /tmp/bench.out 2>&1 &
EOF
            exit 0
            ;;
        *) echo "unknown arg: $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done

# ---- launch -------------------------------------------------------------

echo "============================================================"
echo "hlbench eval matrix"
echo "============================================================"
echo "  envs:          ${ENVS}"
echo "  budget/env:    ${BUDGET}"
echo "  max-turns/env: ${MAX_TURNS}"
echo "  model:         ${MODEL}"
echo "  turn-timeout:  ${TURN_TIMEOUT}s"
echo "  runs-root:     ${RUNS_ROOT}"
echo "  exp-id:        ${EXP_ID}"
echo "  max-parallel:  ${MAX_PARALLEL}"
echo "============================================================"
echo ""

# ENVS is intentionally word-split (un-quoted expansion) because
# run_matrix.py takes --envs as nargs='+'. `exec` so signals
# (Ctrl-C, SIGTERM from nohup-bg) reach python and propagate to its
# hlbench-agent children directly.
exec "${VENV_PY}" "${REPO_ROOT}/scripts/run_matrix.py" \
    --envs ${ENVS} \
    --budget "${BUDGET}" \
    --max-turns "${MAX_TURNS}" \
    --model "${MODEL}" \
    --turn-timeout "${TURN_TIMEOUT}" \
    --runs-root "${RUNS_ROOT}" \
    --exp-id "${EXP_ID}" \
    --max-parallel "${MAX_PARALLEL}"
