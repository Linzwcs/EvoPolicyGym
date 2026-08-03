# EvoPolicyGym InvertedPendulum Benchmark

This independently installable distribution adapts Gymnasium
`InvertedPendulum-v5` through EvoPolicyGym's public authoring SPI and official
packaged MuJoCo model.

## Public interface

```python
from inverted_pendulum import (
    InvertedPendulumBenchmark,
    InvertedPendulumConfig,
    baseline_program,
)

benchmark = InvertedPendulumBenchmark(
    InvertedPendulumConfig(
        frame_skip=2,
        reset_noise_scale=0.01,
    )
)
```

Simulation cadence, actuator gear, termination/reward rules, and reset noise are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official
`inverted_pendulum.xml` model is fixed. Custom XML paths and rendering/camera
settings are Host-owned and never cross the Policy boundary.

## Contract

The Policy receives four unclipped `float64`-derived values: cart position,
pole angle relative to upright, cart linear velocity, and pole angular
velocity. Positive cart position is rightward.

An Action is an exact one-float list `[cart_control]` in `[-3.0, 3.0]`.
Integers, scalar floats, tuples, non-finite values, and out-of-range values are
rejected rather than converted or clipped. The value is a MuJoCo actuator
control, not a force expressed directly in newtons: the official model applies
actuator gear `100`, so feedback reports both requested control and the
gear-scaled generalized force.

The official model timestep is `0.02` seconds. With default `frame_skip=2`,
one Policy step advances `0.04` simulated seconds. The Episode terminates when
`abs(pole_angle) > 0.2` radians (strictly greater) or any state becomes
non-finite, and otherwise truncates after 1000 steps. Every non-terminating
step gives reward `1`; the terminal unhealthy step gives `0`. The scalar score
is mean Episode return, with a maximum of `1000`. Gymnasium's published
solution threshold is `950`.

Policy failure receives a `-1000` return. The packaged baseline applies zero
force and is intentionally weak.

## Feedback and trace

Feedback reports mean return and steps, outcome counts, cart displacement and
velocity extrema, final and maximum pole angle, minimum angle margin to the
failure boundary, pole angular-velocity extrema, action magnitude, Policy
failures, and bounded trace coverage.

`trace.jsonl` retains at most eight Episodes. Every transition contains the
complete before/after observations, exact Action and named component, elapsed
time, remaining steps, requested and gear-scaled control, state extrema,
current and minimum angle margin, reward and cumulative return, health status,
and an explicit terminal reason (`none`, `fallen`, `time_limit`, or both).

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/inverted_pendulum
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
