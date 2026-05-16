#!/usr/bin/env bash
set -euo pipefail

REGION="global"
MODEL="MiniMax-M2.7"
SETTINGS_PATH="${CLAUDE_SETTINGS_PATH:-$HOME/.claude/settings.json}"
DRY_RUN=0
API_KEY_ENV="${API_KEY_ENV:-MINIMAX_API_KEY}"

usage() {
  cat <<'EOF'
Configure Claude Code to use MiniMax M2.7 through Claude's Anthropic-compatible settings.

Usage:
  MINIMAX_API_KEY=<key> ./scripts/configure_claude_minimax.sh [options]

Options:
  --region global|china       Endpoint region. Default: global.
  --model <name>              Model name. Default: MiniMax-M2.7.
  --settings-path <path>      Claude settings path. Default: ~/.claude/settings.json.
  --api-key-env <name>        Environment variable holding the API key. Default: MINIMAX_API_KEY.
  --dry-run                   Print the resulting settings JSON without writing.
  -h, --help                  Show this help.

Notes:
  Shell-level ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL override settings.json.
  Unset them in the current shell before testing:
    unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="${2:?--region requires a value}"
      shift 2
      ;;
    --model)
      MODEL="${2:?--model requires a value}"
      shift 2
      ;;
    --settings-path)
      SETTINGS_PATH="${2:?--settings-path requires a value}"
      shift 2
      ;;
    --api-key-env)
      API_KEY_ENV="${2:?--api-key-env requires a value}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$REGION" in
  global)
    BASE_URL="https://api.minimax.io/anthropic"
    ;;
  china)
    BASE_URL="https://api.minimaxi.com/anthropic"
    ;;
  *)
    echo "--region must be global or china, got: $REGION" >&2
    exit 2
    ;;
esac

API_KEY="${!API_KEY_ENV:-}"
if [[ -z "$API_KEY" && "$DRY_RUN" != "1" ]]; then
  printf "Enter MiniMax API key for Claude Code: " >&2
  stty -echo
  read -r API_KEY
  stty echo
  printf "\n" >&2
fi

if [[ -z "$API_KEY" ]]; then
  API_KEY="<MINIMAX_API_KEY>"
fi

if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" || -n "${ANTHROPIC_BASE_URL:-}" ]]; then
  cat >&2 <<'EOF'
Warning: ANTHROPIC_AUTH_TOKEN or ANTHROPIC_BASE_URL is set in this shell.
Claude Code gives shell environment variables priority over ~/.claude/settings.json.
Run this before testing the new settings:
  unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
EOF
fi

export SETTINGS_PATH BASE_URL API_KEY MODEL DRY_RUN

python3 - <<'PY'
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

settings_path = Path(os.environ["SETTINGS_PATH"]).expanduser()
base_url = os.environ["BASE_URL"]
api_key = os.environ["API_KEY"]
model = os.environ["MODEL"]
dry_run = os.environ["DRY_RUN"] == "1"

if settings_path.exists():
    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{settings_path} is not valid JSON: {exc}") from exc
else:
    settings = {}

if not isinstance(settings, dict):
    raise SystemExit(f"{settings_path} must contain a JSON object")

env = settings.get("env")
if env is None:
    env = {}
elif not isinstance(env, dict):
    raise SystemExit(f"{settings_path} field 'env' must be a JSON object")

env.update(
    {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": api_key,
        "API_TIMEOUT_MS": "3000000",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
    }
)
settings["env"] = env

rendered = json.dumps(settings, indent=2, sort_keys=True) + "\n"
if dry_run:
    print(rendered, end="")
else:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if settings_path.exists():
        backup_path = settings_path.with_suffix(settings_path.suffix + ".bak")
        shutil.copy2(settings_path, backup_path)
    settings_path.write_text(rendered)
    print(f"Updated {settings_path}")
PY

cat <<EOF

Claude Code MiniMax config:
  settings: $SETTINGS_PATH
  base_url: $BASE_URL
  model: $MODEL

Verify in Claude Code:
  claude
  /status
  /model
EOF
