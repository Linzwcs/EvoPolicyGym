# EvoPolicyGym

EvoPolicyGym is benchmark infrastructure for evaluating whether coding agents can
improve executable policies from budget-limited environment feedback. It follows
an online-judge style protocol: an agent edits code in a workspace, submits
policy rollouts through a local API, receives feedback artifacts, and continues
until the episode budget is exhausted.

The current implementation focuses on a local Python harness, reproducible run
artifacts, agent adapters, and Gymnasium-style environment integration.

## Status

EvoPolicyGym is alpha software. The active package is `src/evopolicygym`; frozen
v1 material lives under `archive/v1/` for reference only.

## Install

```bash
uv sync
```

Optional environment families are split into extras:

```bash
uv sync --extra env-gym
uv sync --extra env-compatible
uv sync --extra env-visual
uv sync --extra env-web
uv sync --extra env-jax
uv sync --extra env-mario
```

`env-jax` and `env-mario` are separate runtime targets. Gymnasium's JAX envs
need `numpy>=2.1`, while MO-Gymnasium's Mario extra currently pins
`numpy<2.0`, so they should be tested in separate virtual environments.

Some visual/browser environments need runtime assets. See
[`docs/envs/overview.md`](docs/envs/overview.md) for Atari ROM and MiniWoB++
setup notes.

## Quickstart

Run the unit suite:

```bash
uv run python -m unittest discover -s tests
```

Run the CLI help:

```bash
uv run evopolicygym --help
```

The old `feedbackgym` package and CLI names are not supported.

Create reproducible external case splits:

```bash
uv run evopolicygym data make \
  --env gym/taxi \
  --root data/gym/taxi \
  --seed 0 \
  --train-size 64 \
  --valid-size 64 \
  --heldout-size 256
```

Run a local command-agent benchmark session:

```bash
uv run evopolicygym run \
  --env toy \
  --runs runs \
  --model script \
  --exp-id smoke-001 \
  --budget 8 \
  --agent command -- python agent.py
```

Run from a TOML/JSON config:

```bash
uv run evopolicygym run --config docs/examples/cartpole-codex.toml
```

## Repository Layout

- `src/evopolicygym/`: package source.
- `docs/protocol/`: normative protocol draft.
- `docs/envs/`: environment coverage notes, roadmap, and discovery output.
- `docs/examples/`: small checked-in example configs and fixtures.
- `tests/`: standard-library `unittest` suite.
- `archive/v1/`: frozen legacy code, docs, analysis, and v0 run data.

Generated run outputs belong under `runs/` or `experiment/` and are ignored by
default.

## Agent Adapters

EvoPolicyGym currently includes adapters for:

- generic persistent JSONL command agents;
- OpenAI Codex CLI;
- Claude Code;
- Kimi Code.

Each adapter preserves one logical agent session across benchmark turns while
the server controls rollout budget, feedback artifacts, hidden validation, and
final scoring.

## Environment Discovery

Check the structured EvoPolicyGym environment manifest and run smoke checks for
registered environments:

```bash
uv run evopolicygym check-envs
uv run evopolicygym check-envs --env gym/taxi
uv run evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1
uv run evopolicygym check-envs --discover --min-level L0
```

Regenerate the installed environment registry report with:

```bash
uv run evopolicygym discover-envs \
  --output docs/envs/discovered.json \
  --markdown docs/envs/env_list.md
```

The discovery report reflects installed optional packages; it is not a promise
that every discovered task already has a calibrated EvoPolicyGym scoring setup.

## Safety

EvoPolicyGym executes agent-authored Python policies. Treat benchmark runs as
untrusted code execution. Use sandboxing, isolated workspaces, and disposable
credentials for live agent experiments. See [`SECURITY.md`](SECURITY.md).

## License

EvoPolicyGym is released under the MIT License. See [`LICENSE`](LICENSE).
