# EvoPolicyGym CartPole Benchmark

This directory is the minimal reference for an independently installable
EvoPolicyGym Benchmark. It adapts Gymnasium `CartPole-v1` and depends only on
the public EvoPolicyGym SDK.

The package contains:

- `CartPoleBenchmark`: public specification, deterministic Episode planning,
  Environment construction, scoring, and Feedback;
- `CartPoleEnvironment`: one fresh seeded Gymnasium instance per Episode;
- `baseline_program()`: an intentionally weak initial Policy Program;
- tests for deterministic replay, invalid Actions, Feedback privacy, trace
  publication, and direct Evaluation.

## Feedback

The score is mean Episode return. A Policy failure contributes zero return.
Feedback also contains a compact summary and one public `trace.jsonl` Artifact.
At most eight Episodes are traced so publication remains bounded.

Each trace begins with an Episode record followed by zero or more transition
records. A transition contains:

- the observation received by the Policy;
- the unmodified Action;
- reward and next observation;
- termination and truncation flags.

Environment seeds, Policy seeds, scenarios, Host paths, private metrics, and
runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/cartpole
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation tests is explicitly unsafe and
provides no isolation. The test Programs are trusted package fixtures.

Agent choice, Run coordination, execution settings, workspace management, and
CLI presentation remain outside this Benchmark package.
