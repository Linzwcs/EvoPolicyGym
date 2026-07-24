# Repository Guidelines

## Project Structure

This is a Python 3.12 project managed with `uv`. The active clean-slate package
is under `src/evopolicygym/`:

- `benchmark.py`, `program.py`, `results.py`, and `artifacts.py`: supported
  public values and SDK facades;
- `policy.py`: Policy-author-facing ABI;
- `agents/base.py`: public `CodingAgent`, `AgentTask`, and `AgentInvocation`
  integration template;
- `agents/`: import-safe Agent provider selections; provider implementations
  translate Host-owned tasks without duplicating Run instructions;
- `authoring/`: the only public authoring/conformance surface for external
  Benchmark distributions;
- `evaluation/`: public `EvaluationConfig`/`evaluate()` and private direct
  Evaluation rules;
- `run/`: public `RunConfig`/`run()` plus Session rules, Run records, Feedback
  publication, and Agent Session transport;
- `execution/`: public execution selections; `execution/process/` contains only
  the current Policy and command-Agent process mechanisms;
- `_protocol/`: pure versioned Policy/Agent bytes-to-value codecs; it never
  opens files, sockets, or descriptors;
- `cli.py`: thin presentation over the public SDK.

The accepted package ownership and import graph are defined by
`ARCHITECTURE.md` and enforced by
`tests/test_package_architecture.py`. Do not reintroduce the removed
`_evaluation`, `_evolution`, `_execution`, `_composition`, `_local`, `_engine`,
`_adapters`, `_wire`, `_wiring`, or `settings` namespaces.

Independently installable Benchmark distributions live under `environments/`.
The active CartPole Benchmark is at `environments/cartpole/`. Its
`evopolicygym-benchmark-cartpole==0.1.0` distribution imports as
`cartpole`, requires the base `evopolicygym==0.3.*`, and is
not present in the base wheel. The removed
`evopolicygym._reference.cartpole` module has no compatibility shim. This
package uses only the public authoring SPI and Gymnasium.

Current package behavior and architecture are documented in `README.md` and
`ARCHITECTURE.md`. Superseded implementations and their associated products are
not included in the active repository and are not dependencies or extension
surfaces. Private research and local reference material live under ignored
directories. Generated outputs belong under `runs/`.

The active implementation has no compatibility dependency on earlier Policy,
Runtime, HTTP, artifact, or test contracts. Do not add `legacy`, adapter, or
version-suffixed Python namespaces to recover removed behavior.

## Build and Test Commands

- `uv sync --extra dev`: create or update the Kernel environment.
- `uv run python -m unittest discover -s tests`: run the Kernel tests.
- `uv run ruff check src tests`: run the Kernel linter.
- `uv run mypy`: run strict type checks.
- `uv run evopolicygym --version`: print the package and protocol versions.
- `uv lock`: refresh the Kernel lock after dependency changes.
- `uv build environments/cartpole`: build the independent
  CartPole Benchmark.

`ProcessExecution` is explicitly unsafe for hostile code. The acknowledgement
flag does not provide isolation. Never describe it as a sandbox or silently use
it when a formal virtualization profile was requested.

## Coding Style

Use idiomatic Python with 4-space indentation, `snake_case` functions/modules,
`PascalCase` classes, and concise type hints on public APIs. Domain values are
frozen dataclasses where practical. Keep I/O and subprocess details out of
`evaluation/_service.py`, `run/_session.py`, and `_protocol`. Keep
provider-specific behavior out of `execution/process`; narrow contracts live
beside the service that consumes them.

Policy-visible values must use the bounded PolicyValue ABI. Never use pickle,
custom Python objects, Host paths, file descriptors, credentials, Case identity,
environment seeds, pool identity, or scorer objects across the Policy boundary.

## Testing

Tests use the standard-library `unittest` runner. Name files `test_*.py` and keep
test doubles local unless reused. Changes to runtime semantics must cover both
the typed failure domain and cleanup behavior. Required invariants include:

- a fresh Policy process/instance/scratch for every Episode;
- same-Episode Policy state may persist across `act()` calls;
- invalid Actions are never repaired or replaced;
- Policy failure stops before another `World.step()`;
- Environment and Backend faults do not become Policy penalties;
- Feedback contains no private Case, seed, path, or execution evidence;
- complete malformed guest frames, partial frames, and trusted-input errors keep
  their distinct classifications.

When protocol semantics change, update `README.md`, `ARCHITECTURE.md`, and
representative tests together.

## Generated and External Files

Do not hand-edit generated feedback, logs, checkpoints, discovery reports, or
run-local agent instructions unless the task is explicit artifact repair. Keep
large generated files out of source modules and call them out in reviews.

## Commits and Pull Requests

Use short imperative subjects with an optional scope, such as
`runtime: enforce episode cleanup` or `docs: define policy VM boundary`. Pull
requests should state purpose, affected paths, commands run, and any remaining
security or virtualization limitation.
