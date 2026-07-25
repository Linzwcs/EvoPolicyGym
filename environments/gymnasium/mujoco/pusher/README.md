# EvoPolicyGym Pusher Benchmark

This independently installable distribution adapts Gymnasium `Pusher-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from pusher import PusherBenchmark, PusherConfig, baseline_program

benchmark = PusherBenchmark(
    PusherConfig(
        frame_skip=5,
        reward_near_weight=0.5,
        reward_dist_weight=1.0,
        reward_control_weight=0.1,
    )
)
```

Simulation cadence and reward weights are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `pusher_v5.xml` model
is fixed. Custom XML paths and rendering/camera settings are Host-owned and
never cross the Policy boundary.

## Contract

The Policy receives 23 named floats:

- seven arm joint angles and seven angular velocities;
- fingertip x/y/z position;
- movable object x/y/z position;
- goal x/y/z position.

An Action is an exact seven-float list of shoulder, arm, elbow, forearm, and
wrist torques, each in `[-2.0, 2.0]`. Integers, tuples, non-finite values, and
out-of-range values are rejected rather than converted or clipped.

Pusher does not terminate naturally and truncates after 100 steps. Each reward
is the negative weighted object-goal distance, fingertip-object distance, and
squared control magnitude. The scalar Benchmark score is mean Episode return;
Gymnasium publishes a solution threshold of `0.0`.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, final object-goal distance, Policy
failures, and bounded trace coverage. `trace.jsonl` retains at most eight
Episodes with complete observations, exact Actions, all three reward terms,
and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/pusher
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
