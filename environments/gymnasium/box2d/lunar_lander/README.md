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

Position is centered on the landing pad: `x=0` is horizontally centered,
`y=0` is the pad height adjusted for the deployed-leg offset, and `abs(x)>=1`
is the horizontal viewport boundary. The velocity fields are Gymnasium's
normalized observations; their documented physical conversion factors are
published in the spec. Angle is in radians, while observed angular velocity
multiplies by `2.5` to obtain radians per second.

In discrete mode, Actions are exact integers:

- `0`: do nothing;
- `1`: fire the left orientation engine;
- `2`: fire the main engine;
- `3`: fire the right orientation engine.

In continuous mode, an Action is an exact two-float list
`[main_engine, lateral_engine]`, with each value in `[-1.0, 1.0]`. Out-of-range
Actions are rejected rather than clipped. Main values at or below zero are
off; a value just above zero starts above 50% power, with exact power
`(main_engine + 1) / 2`. The lateral engine is off throughout the inclusive
interval `[-0.5, 0.5]`; values below it fire left and values above it fire right
at power `abs(lateral_engine)`.

The Episode terminates on a crash, leaving the viewport, or a settled landing,
and otherwise truncates at Gymnasium's 1000-step limit. On a normal transition:

```text
position shaping = -100*hypot(x_position, y_position)
velocity shaping = -100*hypot(x_velocity, y_velocity)
angle shaping    = -100*abs(angle)
contact shaping  = +10 per contacting leg
reward = delta(total shaping) - 0.30*main_power - 0.03*side_power
```

A settled landing replaces the entire transition reward with `+100`; a crash
or horizontal viewport exit replaces it with `-100`. These are overrides, not
bonuses added to the normal terms. The scalar Benchmark score is mean Episode
return; Gymnasium's published solution threshold is 200. Physics constants,
normalization factors, engine thresholds, wind/turbulence behavior, and reward
formulas are all published through `environment_parameters`.

Policy failure receives a conservative `-1000` return. The packaged baseline
uses no thrust in either action mode and is intentionally weak.

## Feedback and trace

Feedback distinguishes settled landings, crashes, horizontal viewport exits,
and time limits. It reports closest and final normalized pad distance, speed,
angle error, a combined landing-state penalty, each shaping delta, requested
versus actually charged main/side fuel cost, engine firing fractions, leg
contact fractions, Policy failures, and bounded trace coverage.

`trace.jsonl` retains at most eight Episodes with complete semantic
observations, named exact Actions, rewards, termination flags, and per-step
metrics. Metrics expose exact engine activation, power and impulse scale;
requested and charged fuel; all four reward deltas and terminal override;
cumulative reward decomposition; normalized and recovered world-coordinate
state; contact changes and support state; closest landing-state penalty;
remaining steps; and an exact terminal reason. Episode rows carry the matching
aggregate diagnostics and explicit outcome.

Every Episode with at least one valid transition also preserves every captured
600 × 400 upstream `rgb_array` frame losslessly in `rendered-frames.npz`, with
step indices, rewards, reward presence, and cumulative returns. The Benchmark
captures the initial scene, first result, terminal result, and a fixed stride
of intermediate results, with at most 42 frames per Episode. It derives an
H.264 MP4 at five frames per second from exactly those frames for direct
playback. The NPZ is pixel-exact evidence and the MP4 is a presentation
artifact; both use `retention="bulk"`. Raw RGB tensors are removed from
`trace.jsonl` and never become Policy observations. Rendering failure is
diagnostic and cannot change physics or score; zero-step failures publish an
explicit unavailable manifest.

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
