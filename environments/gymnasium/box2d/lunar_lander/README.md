# EvoPolicyGym LunarLander Benchmark

This independently installable distribution adapts Gymnasium
`LunarLander-v3` through EvoPolicyGym's public authoring SPI.

## Public interface

```python
from lunar_lander import (
    LunarLanderBenchmark,
    LunarLanderConfig,
    baseline_program,
)

benchmark = LunarLanderBenchmark(
    LunarLanderConfig(
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
    )
)
```

All non-rendering Gymnasium constructor parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy.

## Contract

The Policy receives a semantic object with:

- normalized horizontal and vertical position;
- horizontal and vertical velocity;
- angle and angular velocity;
- exact boolean contact state for both landing legs.

In discrete mode, Actions are exact integers:

- `0`: do nothing;
- `1`: fire the left orientation engine;
- `2`: fire the main engine;
- `3`: fire the right orientation engine.

In continuous mode, an Action is an exact two-float list
`[main_engine, lateral_engine]`, with each value in `[-1.0, 1.0]`. Out-of-range
Actions are rejected rather than clipped. Positive main-engine values enable
thrust. Lateral values below `-0.5` fire the left engine and values above
`0.5` fire the right engine.

The Episode terminates on a crash, leaving the viewport, or a settled landing,
and otherwise truncates at Gymnasium's 1000-step limit. Reward combines
position, velocity, angle, leg contact, engine cost, and terminal
crash/landing reward. The scalar Benchmark score is mean Episode return;
Gymnasium's published solution threshold is 200.

Policy failure receives a conservative `-1000` return. The packaged baseline
uses no thrust in either action mode and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, successful landings, Policy failures,
and bounded trace coverage. `trace.jsonl` retains at most eight Episodes with
complete semantic observations, exact Actions, rewards, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/box2d/lunar_lander
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

The independent project installs Gymnasium's `box2d` extra. `ProcessExecution`
used by Evaluation tests is explicitly unsafe and provides no isolation. The
packaged baseline is trusted test code.
