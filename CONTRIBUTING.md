# Contributing

EvoPolicyGym is managed as a Python 3.12 project with `uv`.

## Setup

```bash
uv sync
```

For optional environment work, install only the families you need:

```bash
uv sync --extra env-gym
uv sync --extra env-compatible
uv sync --all-extras
```

## Development Commands

```bash
uv run python -m unittest discover -s tests
uv run evopolicygym --help
uv run evopolicygym discover-envs --output docs/envs/discovered.json --markdown docs/envs/env_list.md
```

If you change dependencies, run:

```bash
uv lock
```

## Code Organization

- Keep core vocabulary and ports in `src/evopolicygym/core` dependency-light.
- Keep concrete filesystem, runtime, HTTP, and subprocess behavior under
  `src/evopolicygym/infra`.
- Keep agent launch/session adapters under `src/evopolicygym/agent`.
- Keep environment registrations under `src/evopolicygym/envs`.
- Update `docs/protocol/` when protocol behavior changes.
- Do not edit `archive/v1/` for active development.

## Tests

Tests use `unittest` and live under `tests/`. Name new tests after the behavior
under test, for example `test_suite.py` or `test_discover.py`.

Prefer focused tests around contracts: schema shape, budget accounting,
artifact layout, environment adapters, and harness lifecycle. Heavy simulator
or browser tests should stay out of the default CI path unless they can run
cheaply and deterministically.

## Generated Artifacts

Do not commit generated run outputs from `runs/`, `experiment/`, local browser
caches, Atari ROMs, or MiniWoB++ clones. Small examples belong in
`docs/examples/`.

## Pull Requests

Include:

- purpose and affected areas;
- commands run;
- protocol or artifact compatibility notes;
- screenshots or run artifact references only when relevant.

Use short imperative commit subjects, for example `docs: clarify artifact layout`
or `envs: add gym discovery report`.
