# Real Codex Smoke

This smoke run checks the live Codex CLI path against the first non-toy
environment. It is intentionally manual because it depends on local Codex
authentication, network access, and account limits.

## Preconditions

- `codex` is installed and authenticated on the machine running the smoke.
- `uv sync` has been run for this repository.
- Port binding to `127.0.0.1` is allowed.
- The Codex harness can access the local EvoPolicyGym HTTP server. The checked-in
  config sets `[agent] bypass = true` because Codex `workspace-write` sandboxing
  may block localhost connections.

## Run

Use the checked-in CartPole config:

```bash
uv run evopolicygym run \
  --config docs/examples/cartpole-codex.toml
```

The config uses `env = "cartpole"`, `budget = 16`, and `maximum = 4`, so the
agent can spend the budget through four submit turns. It does not pin a Codex
model; the local Codex default is used unless you add `model = "..."` under
`[agent]`. The `bypass` flag only changes Codex CLI tool permissions for the
authoring session; policy rollout sandboxing is still controlled by
EvoPolicyGym.

## Expected Result

The command should print JSON with `done: true`, `reason: "done"`, and
`submits: 4`. The run artifacts are written under
`runs/codex/cartpole/real-codex-smoke/`, including `run.json`,
`workspace/feedback/`, `checkpoints/`, `logs/codex_turns/`, and the staged
`workspace/AGENTS.md`.

After the run, validate artifacts with:

```bash
uv run python - <<'PY'
from evopolicygym.check import check
report = check('runs/codex/cartpole/real-codex-smoke')
print(report.ok)
for issue in report.issues:
    print(issue.code, issue.path, issue.message)
PY
```

Keep this run out of automated tests unless the Codex CLI is replaced with a
fake binary, as in `tests/test_cli.py`.
