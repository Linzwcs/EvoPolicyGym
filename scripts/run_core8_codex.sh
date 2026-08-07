#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repository_root}"

run_seed="${CORE8_RUN_SEED:-20260807}"
run_root="${CORE8_RUN_ROOT:-runs/core8-v1}"
models=(
  gpt-5.6-luna
  gpt-5.6-terra
  gpt-5.6-sol
)

mkdir -p "${run_root}"

run_environment() {
  local project="$1"
  local module="$2"
  local benchmark_factory="$3"
  local environment_slug="$4"
  local assessment_episodes="$5"
  local episode_timeout="$6"

  for model in "${models[@]}"; do
    local lane="${model##*-}"
    local record_base="${run_root}/${environment_slug}-${lane}-seed-${run_seed}"
    local record="${record_base}"
    local attempt=1

    while [[ -e "${record}" ]]; do
      if [[ -f "${record}/run.json" ]]; then
        echo "Skipping completed Run record: ${record}"
        continue 2
      fi
      attempt=$((attempt + 1))
      record="${record_base}-attempt-${attempt}"
    done

    if (( attempt > 1 )); then
      echo "Retaining incomplete Run record; retrying at ${record}" >&2
    fi

    echo "Running ${environment_slug} / ${model}"

    uv run --project "${project}" \
      python scripts/run_benchmark_codex.py \
      --module "${module}" \
      --benchmark-factory "${benchmark_factory}" \
      --baseline-factory baseline_program \
      --model "${model}" \
      --reasoning-effort xhigh \
      --record-to "${record}" \
      --seed "${run_seed}" \
      --episode-budget 128 \
      --episode-pool-size 128 \
      --max-episodes-per-submission 32 \
      --assessment-split test \
      --assessment-episodes "${assessment_episodes}" \
      --episode-timeout-seconds "${episode_timeout}" \
      --agent-timeout-seconds 7200 \
      --progress plain \
      --allow-unsafe-process
  done
}

# Validation is intentionally omitted. The Agent hands off exactly one final
# candidate, and Assessment evaluates that selected Program directly.

run_environment \
  environments/minigrid/minigrid/keycorridor \
  minigrid_keycorridor \
  KeyCorridorBenchmark \
  keycorridor-s4r3 \
  128 \
  60

run_environment \
  environments/gymnasium/mujoco/half_cheetah \
  half_cheetah \
  HalfCheetahBenchmark \
  half-cheetah-v5 \
  64 \
  120

run_environment \
  environments/gymnasium/box2d/car_racing \
  car_racing \
  CarRacingBenchmark \
  car-racing-v3 \
  64 \
  120

run_environment \
  environments/atcoder/ahc054/treants_forest \
  treants_forest \
  TreantsForestBenchmark \
  treants-forest \
  64 \
  120
