# EvoPolicyGym InvertedDoublePendulum Benchmark

This independently installable distribution adapts Gymnasium
`InvertedDoublePendulum-v5` through EvoPolicyGym's public authoring SPI and
official packaged MuJoCo model.

## Public interface

```python
from inverted_double_pendulum import (
    InvertedDoublePendulumBenchmark,
    InvertedDoublePendulumConfig,
    baseline_program,
)

benchmark = InvertedDoublePendulumBenchmark(
    InvertedDoublePendulumConfig(
        frame_skip=5,
        healthy_reward=10.0,
        reset_noise_scale=0.1,
    )
)
```

Simulation cadence, healthy reward, and reset noise are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official
`inverted_double_pendulum.xml` model is fixed. Custom XML paths and
rendering/camera settings are Host-owned and never cross the Policy boundary.

## Contract

The Policy receives nine named floats:

- cart position;
- sine and cosine of both pole angles;
- cart velocity and both pole angular velocities;
- the cart constraint force, clipped by Gymnasium to `[-10, 10]`.

An Action is an exact one-float list `[cart_force]` in `[-1.0, 1.0]`.
Integers, scalar floats, tuples, non-finite values, and out-of-range values are
rejected rather than converted or clipped.

The Episode terminates when the free tip of the second pole falls to a height
of one meter or less, and otherwise truncates after 1000 steps. Reward is the
configured healthy reward minus tip-distance and angular-velocity penalties.
The scalar Benchmark score is mean Episode return. Gymnasium's published
solution threshold is `9100`.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero force and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, full-horizon balances, Policy
failures, and bounded trace coverage. `trace.jsonl` retains at most eight
Episodes with complete observations, exact Actions, all reward terms, and
termination flags.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/inverted_double_pendulum
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
