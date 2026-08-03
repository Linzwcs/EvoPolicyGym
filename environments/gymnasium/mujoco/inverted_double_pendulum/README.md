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
- sine and cosine of pole 1's angle and pole 2's angle relative to pole 1;
- cart velocity and both pole angular velocities;
- the cart constraint force, clipped by Gymnasium to `[-10, 10]`.

All three velocities are also clipped to `[-10, 10]` in the observation, while
the velocity reward penalty uses the un-clipped MuJoCo angular velocities.
Feedback marks steps that reach the visible clipping limit.
The reward and termination use MuJoCo `site_xpos` for the free tip. Feedback
also reconstructs tip position from the visible qpos trigonometry and reports
their difference, because MuJoCo's derived-site update timing can make the two
values differ slightly after dynamic actions.

An Action is an exact one-float list `[cart_control]` in `[-1.0, 1.0]`.
The official slider actuator has gear `500`; both requested control and the
gear-scaled force are published.
Integers, scalar floats, tuples, non-finite values, and out-of-range values are
rejected rather than converted or clipped.

The Episode terminates when the free tip of the second pole falls to a height
of one meter or less, and otherwise truncates after 1000 steps. Reward is the
configured healthy reward minus tip-distance and angular-velocity penalties.
Both poles are `0.6m`, so the physical maximum tip height is `1.2m`. The pinned
reward nevertheless targets `2.0m`; even a perfectly upright centered pose
therefore receives a fixed `-0.64` distance term. Each Policy step advances
`0.05s` by default (`0.01s` model timestep times frame skip `5`).
The scalar Benchmark score is mean Episode return. Gymnasium's published
solution threshold is `9100`.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero force and is intentionally weak.

## Feedback and trace

Feedback reports mean return and steps; fall/full-horizon outcomes; current,
minimum, and maximum tip height; termination-height safety margin; horizontal
tip deviation; reconstructed absolute and relative pole angles; visible
angular-velocity saturation; cart travel; action use; complete current and
cumulative reward decomposition; Policy failures; and bounded trace coverage.
`trace.jsonl` retains at most eight Episodes with complete named observations,
exact Actions and gear-scaled diagnostics, next observations, and explicit
terminal reason.

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
