# EvoPolicyGym Reacher Benchmark

This independently installable distribution adapts Gymnasium `Reacher-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from reacher import ReacherBenchmark, ReacherConfig, baseline_program

benchmark = ReacherBenchmark(
    ReacherConfig(
        frame_skip=2,
        reward_dist_weight=1.0,
        reward_control_weight=1.0,
    )
)
```

Simulation cadence, actuator gears, target sampling, arm geometry, and reward weights are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `reacher.xml` model is
fixed. Custom XML paths and rendering/camera settings are Host-owned and never
cross the Policy boundary.

## Contract

The Policy receives ten named `float64`-derived values:

- sine and cosine of the shoulder angle and the elbow angle relative to the
  first link;
- target x/y position;
- angular velocity of both joints;
- fingertip-minus-target x/y displacement.

An Action is an exact two-float list containing shoulder and elbow actuator
controls, each in `[-1.0, 1.0]`. Both official actuators apply gear `200`, so
these values are not direct torques in newton-meters. Feedback reports both
requested controls and gear-scaled generalized torques. Integers, tuples,
non-finite values, and out-of-range values are rejected rather than converted
or clipped.

The target is sampled uniformly in x/y from `[-0.2, 0.2]` and rejected unless
its radius is below `0.2` meters. The model timestep is `0.01` seconds; default
`frame_skip=2` advances `0.02` simulated seconds per Policy step. Reacher never
terminates naturally and truncates after 50 steps. Each reward is the negative
weighted fingertip distance minus the weighted squared action norm. The scalar
Benchmark score is mean Episode return; Gymnasium's published solution
threshold is `-3.75`.

Policy failure receives a `-1000` return. The packaged baseline applies zero
torque and is intentionally weak.

## Feedback and trace

Feedback reports initial, final, best, and worst fingertip-target distance;
final distance reduction; the closest-approach step; angular-velocity extrema;
action effort; cumulative distance and control rewards; outcome counts; Policy
failures; and bounded trace coverage.

`trace.jsonl` retains at most eight Episodes with complete observations, exact
named controls, gear-scaled torques, reconstructed joint angles and fingertip
coordinates, unit-circle consistency, current and running distances, timing,
reward decomposition, cumulative return, and explicit terminal reason.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/reacher
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
