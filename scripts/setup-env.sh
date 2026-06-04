#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MINIWOB_COMMIT="7fd85d71a4b60325c6585396ec4f48377d049838"
MINIWOB_DIR="third_party/miniwob-plusplus"

usage() {
  cat <<'EOF'
Usage: scripts/setup-env.sh [options]

Installs EvoPolicyGym dependencies for local benchmark runs.

Default behavior installs the primary Core-16 stack:
  - base package
  - dev tools
  - env-gym
  - env-compatible

Options:
  --smoke          Install only base + dev dependencies for toy/cartpole smoke.
  --core           Install Core-16 dependencies. This is the default.
  --web            Also install BrowserGym MiniWoB++ dependencies and assets.
  --visual         Also install env-visual.
  --heavy          Also install env-heavy.
  --multi          Also install env-multi.
  --jax            Install env-jax instead of the Core-16 stack.
  --mario          Install env-mario instead of the Core-16 stack.
  --atari-roms     Install Atari ROMs with AutoROM after env-gym is installed.
  --no-dev         Do not install the dev extra.
  -h, --help       Show this help.

Notes:
  env-jax and env-mario are intentionally separate because their NumPy
  requirements conflict. Run them in separate virtual environments.
EOF
}

mode="core"
with_dev=1
with_web=0
with_visual=0
with_heavy=0
with_multi=0
with_roms=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      mode="smoke"
      ;;
    --core)
      mode="core"
      ;;
    --web)
      with_web=1
      ;;
    --visual)
      with_visual=1
      ;;
    --heavy)
      with_heavy=1
      ;;
    --multi)
      with_multi=1
      ;;
    --jax)
      mode="jax"
      ;;
    --mario)
      mode="mario"
      ;;
    --atari-roms)
      with_roms=1
      ;;
    --no-dev)
      with_dev=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

extras=()
if [[ "$with_dev" -eq 1 ]]; then
  extras+=("dev")
fi

case "$mode" in
  smoke)
    ;;
  core)
    extras+=("env-gym" "env-compatible")
    ;;
  jax)
    extras+=("env-jax")
    ;;
  mario)
    extras+=("env-mario")
    ;;
  *)
    echo "unsupported mode: $mode" >&2
    exit 2
    ;;
esac

if [[ "$with_web" -eq 1 ]]; then
  extras+=("env-web")
fi
if [[ "$with_visual" -eq 1 ]]; then
  extras+=("env-visual")
fi
if [[ "$with_heavy" -eq 1 ]]; then
  extras+=("env-heavy")
fi
if [[ "$with_multi" -eq 1 ]]; then
  extras+=("env-multi")
fi

cmd=(uv sync)
for extra in "${extras[@]}"; do
  cmd+=(--extra "$extra")
done

echo "+ ${cmd[*]}"
"${cmd[@]}"

if [[ "$with_web" -eq 1 ]]; then
  mkdir -p third_party
  if [[ ! -d "$MINIWOB_DIR/.git" ]]; then
    echo "+ git clone https://github.com/Farama-Foundation/miniwob-plusplus.git $MINIWOB_DIR"
    git clone https://github.com/Farama-Foundation/miniwob-plusplus.git "$MINIWOB_DIR"
  fi

  if [[ -n "$(git -C "$MINIWOB_DIR" status --porcelain)" ]]; then
    echo "$MINIWOB_DIR has local changes; refusing to change its checkout." >&2
    exit 1
  fi

  echo "+ git -C $MINIWOB_DIR fetch --tags origin"
  git -C "$MINIWOB_DIR" fetch --tags origin
  echo "+ git -C $MINIWOB_DIR checkout $MINIWOB_COMMIT"
  git -C "$MINIWOB_DIR" checkout "$MINIWOB_COMMIT"

  echo "+ uv run python -m playwright install chromium"
  uv run python -m playwright install chromium

  miniwob_url="$(python - <<'PY'
from pathlib import Path
print((Path("third_party/miniwob-plusplus/miniwob/html/miniwob").resolve()).as_uri() + "/")
PY
)"
  echo "MiniWoB++ HTML ready at: $miniwob_url"
  echo "Set this if you run from outside the repository root:"
  echo "  export MINIWOB_URL=$miniwob_url"
fi

if [[ "$with_roms" -eq 1 ]]; then
  rom_dir="$(uv run python - <<'PY'
from pathlib import Path
import site
print(Path(site.getsitepackages()[0]) / "ale_py" / "roms")
PY
)"
  echo "+ uv run AutoROM --accept-license --install-dir $rom_dir"
  uv run AutoROM --accept-license --install-dir "$rom_dir"
fi

echo "Environment setup complete."
