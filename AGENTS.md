# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python 3.12 project managed with `uv`. The current executable stub is `main.py`; expand reusable code into a package directory before adding substantial logic. Project metadata and dependency locks live in `pyproject.toml` and `uv.lock`.

Active documentation lives under `docs/`, with protocol chapters in `docs/protocol/` and environment planning in `docs/envs/`. Frozen v1 code, docs, analysis, and v0 experiment data are archived under `archive/v1/` and should not be edited for ongoing EvoPolicyGym work. Private research and paper materials live under ignored `private/` and are not part of the open-source package. Real benchmark case data should live in a configured data directory, not under `src`. `runs/` contains generated benchmark runs organized as `runs/<model>/<env>/<exp-id>/`; treat it as data output unless intentionally curating artifacts.

## Build, Test, and Development Commands

- `uv sync`: create or update the local environment from `pyproject.toml` and `uv.lock`.
- `uv run python main.py`: run the current smoke-test entry point.
- `uv run python -m unittest discover -s tests`: run the current unit tests.
- `uv run evopolicygym run --env toy --runs runs --model script --exp-id smoke-001 --budget 8 --agent command -- python agent.py`: run one local benchmark session with a JSONL command harness under `runs/script/toy/smoke-001/`.
- `uv run evopolicygym data make --env gym/taxi --root data/gym/taxi --seed 0 --train-size 64 --valid-size 64 --heldout-size 256`: create deterministic external case splits for a seed-backed environment.
- Add `--data path/to/data` or `data = "path/to/data"` in `[run]` to load external `train.json`, `valid.json`, and `heldout.json` case splits.
- Add `--retries N` or `[agent] retries = N` to retry harness/service timeouts and exceptions; `retry_backoff` controls exponential backoff seconds.
- `uv run evopolicygym run --config run.toml`: run the same path from a JSON/TOML run spec.
- `uv run evopolicygym suite --config suite.toml`: run a suite and write `suite.json`; set `[suite] jobs = N` for parallel run execution.
- `uv run evopolicygym check-envs --env toy`: check manifest status and smoke-test registered environments.
- `uv run evopolicygym discover-envs --output docs/envs/discovered.json --markdown docs/envs/env_list.md`: regenerate the installed environment discovery report and human-readable checklist.
- In run specs, use `[agent] kind = "codex"` to run through the Codex CLI adapter; omit `binary` to use `codex` from `PATH`. For live Codex runs that must reach the local HTTP API, set `bypass = true`.
- Use `[agent] kind = "claude"` for the Claude Code adapter; configure `model`, `permission`, `tools`, and passthrough `args` only when the default wrapper behavior is insufficient.
- Use `[agent] kind = "kimi"` for the Kimi Code adapter; configure `model` and passthrough `args` only when the default wrapper behavior is insufficient.
- `uv lock`: refresh `uv.lock` after dependency changes.

The package build and `evopolicygym` console script are configured in `pyproject.toml`. There is still no configured formatter, type checker, or pytest dependency; add those explicitly before documenting them as required workflows.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and concise type hints on public APIs. Keep generated run artifacts out of source modules. For protocol docs, preserve the versioned structure and use stable names such as `submit_NNN`, `ep_<XXX>`, and `run.json` consistently.

## Testing Guidelines

Tests currently use the standard-library `unittest` runner under `tests/`. Name files like `test_judge.py`, keep test doubles local to the test module unless reused, and prefer focused unit tests for schema helpers and benchmark lifecycle behavior. When changing protocol semantics, add or update fixtures that validate representative `run.json` and feedback layouts.

## Commit & Pull Request Guidelines

This repository has no local commit history yet, so no enforced message convention exists. Use short, imperative subjects with an optional scope, for example `docs: clarify v2 artifact layout` or `python: add smoke test`.

Pull requests should include a brief purpose statement, affected paths, and commands run. Link related issues or experiment IDs when applicable. Include screenshots only for visual artifact changes, and call out any large generated files added under `runs/`.

## Agent-Specific Instructions

Do not hand-edit generated `workspace/feedback/`, `logs/`, `checkpoints/`, or run-local `workspace/AGENTS.md` unless the task is explicitly artifact repair or analysis. The agent-facing rules source lives under `src/evopolicygym/protocol/AGENTS.md`; update that source and matching `docs/protocol/` notes when protocol rules change.
