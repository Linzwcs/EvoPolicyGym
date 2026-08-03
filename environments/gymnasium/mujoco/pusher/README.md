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

Simulation cadence, actuator gears, reward formulas, reset geometry, and reward weights are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `pusher_v5.xml` model
is fixed. Custom XML paths and rendering/camera settings are Host-owned and
never cross the Policy boundary.

## Contract

The Policy receives 23 unclipped `float64`-derived values:

- seven arm joint angles and seven angular velocities;
- fingertip x/y/z position;
- movable object x/y/z position;
- goal x/y/z position.

An Action is an exact seven-float list of shoulder, arm, elbow, forearm, and
wrist torques in newton-meters, each in `[-2.0, 2.0]`. The official actuators
all have gear `1`, so controls are used directly. Integers, tuples, non-finite
values, and out-of-range values are rejected rather than converted or clipped.

The model timestep is `0.01` seconds; default `frame_skip=5` advances `0.05`
simulated seconds per Policy step. Pusher never terminates naturally and
truncates after 100 steps. Each reward is the negative weighted object-goal
distance, fingertip-object distance, and squared action norm, all measured
after the physics step. The scalar Benchmark score is mean Episode return;
Gymnasium publishes a solution threshold of `0.0`.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback separates reaching from pushing. It reports initial, final, best, and
worst object-goal distance; distance reduction and remaining fraction; final
and minimum fingertip-object distance; object displacement; action effort;
cumulative distance, near, and control rewards; outcome counts; Policy
failures; and bounded trace coverage. This makes it visible when a Policy
improves reward only by bringing the fingertip closer without moving the
object toward the goal.

`trace.jsonl` retains at most eight Episodes with complete observations, exact
named torque components, timing and remaining horizon, displacement and
direction vectors, running distance extrema, action magnitude, all current and
cumulative reward terms, and explicit terminal reason.

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
