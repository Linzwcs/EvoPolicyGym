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

Simulation cadence and reset noise are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official
`inverted_pendulum.xml` model is fixed. Custom XML paths and rendering/camera
settings are Host-owned and never cross the Policy boundary.

## Contract

The Policy receives four named floats: cart position, pole angle, cart linear
velocity, and pole angular velocity.

An Action is an exact one-float list `[cart_force]` in `[-3.0, 3.0]`.
Integers, scalar floats, tuples, non-finite values, and out-of-range values are
rejected rather than converted or clipped.

The Episode terminates when the pole angle exceeds `0.2` radians or any state
becomes non-finite, and otherwise truncates after 1000 steps. Every healthy
step gives reward `1`; the terminal unhealthy step gives `0`. The scalar
Benchmark score is mean Episode return, with a maximum of `1000`. Gymnasium's
published solution threshold is `950`.

Policy failure receives a `-1000` return. The packaged baseline applies zero
force and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, full-horizon balances, Policy
failures, and bounded trace coverage. `trace.jsonl` retains at most eight
Episodes with complete observations, exact Actions, survival reward, and
termination flags.

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
