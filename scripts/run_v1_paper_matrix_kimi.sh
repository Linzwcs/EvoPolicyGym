#!/usr/bin/env bash
# scripts/run_v1_paper_matrix_kimi.sh — v1 paper 16-env matrix runner
# (server-friendly, Moonshot Kimi Code backend).
#
# Same env roster, parallelism shape, and run-dir layout as
# scripts/run_v1_paper_matrix.sh and the codex twin
# (scripts/run_v1_paper_matrix_codex.sh) — only the agent driver
# differs. Drives ``hlbench agent --backend kimi``. All three
# launchers can run side-by-side under different `MODEL_SLUG`s
# without clobbering each other's run dirs, which makes building the
# paper Table 1 (Sonnet vs Codex vs Kimi columns) a one-shot job.
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
#   # default: kimi-k2 × budget=64 × max-turns=8 × parallel=4
#   bash scripts/run_v1_paper_matrix_kimi.sh
#
#   # alternative kimi model (must match a model your kimi-cli is
#   # configured to call — `kimi login` populates the model registry)
#   bash scripts/run_v1_paper_matrix_kimi.sh --model kimi-k2
#
#   # heavier eval (paper-grade)
#   bash scripts/run_v1_paper_matrix_kimi.sh --budget 256 --max-turns 24
#
#   # different parallelism
#   bash scripts/run_v1_paper_matrix_kimi.sh --max-parallel 8
#
#   # nohup background (survives SSH disconnect)
#   nohup bash scripts/run_v1_paper_matrix_kimi.sh > nohup-kimi.out 2>&1 &
#
# Env-var fallbacks (let cron / CI set without flags):
#   MODEL=kimi-k2 BUDGET=256 MAX_TURNS=24 \
#       bash scripts/run_v1_paper_matrix_kimi.sh
#
# Auth: kimi needs a managed Moonshot provider configured via
# ``kimi login`` (populates ``~/.kimi-code/config.toml``). The
# harness inherits whatever the local ``kimi`` CLI is configured
# with — no extra credentials to plumb through.

set -euo pipefail

# ===================================================================
# v1 paper Table 1 — 16 envs across 4 Gymnasium categories.
# Identical roster to run_v1_paper_matrix.sh / _codex.sh so
# cross-backend numbers are directly comparable in paper/table.md.
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
#   MODEL          sonnet           → kimi-k2
#   MODEL_SLUG     claude-code-auto → kimi-auto
#   EXP_ID prefix  v1paper-<model>  → v1paper-kimi-<model>
#   TURN_TIMEOUT   900s (kimi's first-call latency is similar to codex)
# ===================================================================
MODEL="${MODEL:-kimi-k2}"
MODEL_SLUG="${MODEL_SLUG:-kimi-auto}"
BUDGET="${BUDGET:-64}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TURN_TIMEOUT="${TURN_TIMEOUT:-900}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
EXP_ID="${EXP_ID:-v1paper-kimi-${MODEL}-$(date +%Y%m%d-%H%M%S)}"
KIMI_BINARY="${KIMI_BINARY:-kimi}"

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

if ! command -v "${KIMI_BINARY}" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: '${KIMI_BINARY}' (Moonshot Kimi Code CLI) binary not on PATH.

Install:
    # See https://moonshotai.github.io/kimi-code/ for the official
    # install script. It typically lands at ~/.kimi-code/bin/kimi and
    # adds the dir to PATH via your shell rc.

Verify:
    ${KIMI_BINARY} --version    # expect 0.6.0+

Auth:
    ${KIMI_BINARY} login         # interactive, populates
                                 # ~/.kimi-code/config.toml

If your kimi binary lives elsewhere, point the script at it:
    KIMI_BINARY=/path/to/kimi bash scripts/run_v1_paper_matrix_kimi.sh
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
        --kimi-binary)   shift; KIMI_BINARY="$1" ;;
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
    v1paper-kimi-kimi-k2-*)
        if [ "${MODEL}" != "kimi-k2" ]; then
            EXP_ID="v1paper-kimi-${MODEL}-$(date +%Y%m%d-%H%M%S)"
        fi
        ;;
esac

# ===================================================================
# Announce.
# ===================================================================
N_ENVS=${#V1_PAPER_ENVS[@]}

cat <<EOF
============================================================
hlbench-pro — v1 paper Table 1 matrix runner (Kimi backend)
============================================================
  backend:       kimi (Moonshot Kimi Code CLI)
  kimi binary:   ${KIMI_BINARY} ($(${KIMI_BINARY} --version 2>/dev/null | head -1))
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
  60-120 min for default budget=64/max-turns=8 at parallel=4. Kimi's
  first-call latency on a cold machine can add a minute or two per env.

Cost: Kimi 0.6's stream-json doesn't surface per-turn token counts,
so this script can't estimate cost mid-flight. Settle up against
your Moonshot billing dashboard. kimi-k2 (2026-05 pricing) is
generally in the same ballpark as gpt-5-codex; a budget=64 /
max-turns=8 run lands somewhere around \$0.30-\$1.50 per env.

Side note: kimi persists per-session state at
~/.kimi-code/sessions/wd_<basename>_<hash>/session_<uuid>/ outside
the run dir. The harness scrapes the session_id from kimi's
stream-json (primary) or falls back to ~/.kimi-code/session_index.jsonl
filtered by workDir (we give each run a unique workspace_dir, so
the filter is exact). Don't move or clean those directories during
a live matrix — they're load-bearing for resume.

EOF

# ===================================================================
# Launch. exec so signals (Ctrl-C, SIGTERM from nohup-bg) reach
# python and propagate to its hlbench-agent children.
# ===================================================================
exec "${VENV_PY}" "${REPO_ROOT}/scripts/run_matrix.py" \
    --envs "${V1_PAPER_ENVS[@]}" \
    --backend kimi \
    --kimi-binary "${KIMI_BINARY}" \
    --budget "${BUDGET}" \
    --max-turns "${MAX_TURNS}" \
    --model "${MODEL}" \
    --model-slug "${MODEL_SLUG}" \
    --turn-timeout "${TURN_TIMEOUT}" \
    --runs-root "${RUNS_ROOT}" \
    --exp-id "${EXP_ID}" \
    --max-parallel "${MAX_PARALLEL}"
