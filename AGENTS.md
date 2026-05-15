# Repository Guidelines

## Project Structure & Module Organization

This repository uses a Python `src/` layout. Core package code lives under
`src/hlbench/`: `rollout/` runs candidate policies and writes artifacts,
`harness/` creates learner workspaces and orchestrates Codex runs, and
`adapters/` contains environment integrations such as Gymnasium MiniGrid.
Scenario definitions live in `src/hlbench/scenarios/minigrid_*`; each scenario
keeps its `scenario.json`, baseline `policy.py`, task spec, and local README
together. Design notes and benchmark plans live in `docs/`. Generated run
outputs are written to `runs/` or `src/runs/` depending on the working
directory; keep those artifacts out of source changes unless explicitly needed.

## Build, Test, and Development Commands

Run commands from the repository root with `PYTHONPATH=src` unless the package
has been installed in editable mode.

- `PYTHONPATH=src python -m compileall src/hlbench`: checks Python syntax.
- `PYTHONPATH=src python -m hlbench.rollout.run_policy --scenario minigrid_doorkey --split train --episodes 2 --run-id smoke`: runs a DoorKey smoke rollout.
- `PYTHONPATH=src python -m hlbench.rollout.run_policy --scenario minigrid_keycorridor --split train --episodes 1 --run-id keycorridor_smoke`: runs the KeyCorridor smoke scenario.
- `PYTHONPATH=src python -m hlbench.harness.run_codex_step --scenario minigrid_doorkey --run-id codex_skip --skip-codex`: validates evaluator and logging without invoking Codex.

MiniGrid execution requires `gymnasium` and `minigrid`; missing dependency
errors should be fixed in the environment, not worked around in scenario code.

## Coding Style & Naming Conventions

Use standard Python 3 style: four-space indentation, type hints where they make
interfaces clearer, `snake_case` for functions and variables, and `PascalCase`
for classes. Keep scenario directories named `minigrid_<task>` and expose a
`Policy` class from every scenario `policy.py`. Prefer `pathlib.Path`, dataclass
models, and JSON parsing through the standard library, matching the existing
code.

## Testing Guidelines

There is no dedicated test suite in this checkout. Treat `compileall` plus a
small rollout as the minimum verification for code changes. For harness changes,
also run `run_codex_step --skip-codex` to confirm artifact creation and scoring
paths. Name future tests `test_<module>.py` and mirror the package layout under
a top-level `tests/` directory.

## Commit & Pull Request Guidelines

This workspace does not include Git metadata, so no local history conventions
are available. Use concise, imperative commit messages such as
`Add KeyCorridor smoke validation`. Pull requests should describe the scenario
or harness behavior changed, list verification commands run, and call out any
new generated artifacts or dependency assumptions.
