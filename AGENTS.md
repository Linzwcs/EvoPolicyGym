# Repository Guidelines

## Project Structure & Module Organization

This repository uses a Python `src/` layout. Core package code lives under
`src/hlbench/`: `core/` defines scenario, seed, task, and artifact contracts;
`envs/` contains environment backends such as Gymnasium; `rollout/` runs policy
episodes; `harness/` manages workspaces, submissions, evaluation, and scoring;
`workspace/` renders agent-facing files. Scenario definitions live in
`src/hlbench/scenarios/<name>/` with `scenario.json`, baseline `policy.py`,
`task_spec.md`, and `README.md`. Tests live in `tests/`. Generated outputs go
under `runs/<model>/<scenario>/<run_id>/` and should not be committed.

## Build, Test, and Development Commands

Run commands from the repository root with `PYTHONPATH=src` unless the package
has been installed in editable mode.

- `PYTHONPATH=src python -m compileall src/hlbench`: checks Python syntax.
- `PYTHONPATH=src python -m unittest discover -s tests`: runs the regression suite.
- `PYTHONPATH=src python -m hlbench scenario validate --scenario acrobot_swingup`: validates a scenario contract and Gymnasium smoke step.
- `PYTHONPATH=src python -m hlbench run --scenario mountain_car --preset smoke --epochs 1`: runs a one-epoch harness smoke test.
- `PYTHONPATH=src python -m hlbench run --scenario mountain_car --agent-backend command --agent-preset none --preset smoke --epochs 1`: runs explicitly with the no-op command agent.
- `PYTHONPATH=src python -m hlbench rollout --workspace <workspace> --split train --episodes 5 --output-dir experiments/probe`: runs an agent-visible train rollout.

## Coding Style & Naming Conventions

Use standard Python 3 style: four-space indentation, type hints where they clarify
interfaces, `snake_case` for functions and variables, and `PascalCase`
for classes. Scenario directories use descriptive `snake_case` names such as
`mountain_car` or `acrobot_swingup`; every scenario exposes a `Policy` class in
`policy.py`. Prefer `pathlib.Path`, dataclasses, explicit JSON records, and
small functions over framework-heavy abstractions.

## Testing Guidelines

Use `unittest`; name files `tests/test_<area>.py`. Core protocol tests should
cover artifact layout, workspace visibility boundaries, private split behavior,
and minimum-score handling. For scenario changes, run the validator and at least
one smoke harness run. Validation and heldout evaluations must never create
replay, episode, or failure-detail artifacts.

## Commit & Pull Request Guidelines

Use concise, imperative commit messages such as `Add Acrobot scenario` or
`Tighten workspace feedback boundary`. Pull requests should describe behavior
changes, list verification commands, mention new scenarios or dependencies, and
call out any intentional changes to run artifact layout or workspace visibility.
