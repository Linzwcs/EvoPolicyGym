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

## Contract

The live observation is a four-element Python-float list containing cart
position (m), cart velocity (m/s), pole angle from upright (rad), and pole
angular velocity (rad/s). Gymnasium produces the values as `float32`; the
adapter converts them to exact Python `float` carriers without claiming extra
source precision.

Action `0` applies 10 N left and Action `1` applies 10 N right. Dynamics use a
0.02-second explicit-Euler step. Every transition, including a failing one,
rewards `+1`, so return equals survived steps. Absolute cart position above
2.4 m or pole angle above 12 degrees terminates; surviving 500 steps truncates
successfully. Physics constants, initial-state sampling, units, thresholds,
reward, and horizon are public environment parameters.

## Feedback

The score is mean Episode return. A Policy failure contributes zero return.
Feedback also reports mean steps, time-limit successes, cart-limit and
pole-angle-limit terminations, plus one public `trace.jsonl` Artifact. At most
eight Episodes are traced so publication remains bounded.

Each trace begins with an Episode record followed by zero or more transition
records. A transition contains:

- the observation received by the Policy;
- the unmodified Action;
- reward and next observation;
- termination and truncation flags;
- named action and applied force;
- elapsed simulation time, threshold margins, angle in radians and degrees,
  balance flags, survival fraction, and exact terminal reason.

Environment seeds, Policy seeds, scenarios, Host paths, private metrics, and
runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/classic_control/cartpole
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
