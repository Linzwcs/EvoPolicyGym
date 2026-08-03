# EvoPolicyGym Walker2d Benchmark

This independently installable distribution adapts Gymnasium `Walker2d-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from walker2d import Walker2dBenchmark, Walker2dConfig, baseline_program

benchmark = Walker2dBenchmark(
    Walker2dConfig(
        frame_skip=4,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.001,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.8, 2.0),
        healthy_angle_range=(-1.0, 1.0),
        reset_noise_scale=0.005,
        exclude_current_positions_from_observation=True,
    )
)
```

All listed parameters plus cadence, actuator gears, reward formula, velocity clipping, and horizon are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official
`walker2d_v5.xml` model is fixed. Custom XML paths and rendering/camera
settings are Host-owned and never cross the Policy boundary.

## Contract

The default Policy observation contains seventeen named `float64`-derived values: torso height
and angle, six right/left leg joint angles, torso x/z velocity, and seven
angular velocities. All nine velocities are clipped to `[-10, 10]`. With
`exclude_current_positions_from_observation=False`, `torso_x_position` is
prepended.

An Action is an exact six-float list containing right and left thigh, leg, and
foot actuator controls, each in `[-1.0, 1.0]`. All official actuators apply
gear `100`, so controls are not direct torques in newton-meters. Feedback
reports both requested and gear-scaled values. Integers, tuples, non-finite
values, and out-of-range values are rejected rather than converted or clipped.

The model timestep is `0.002` seconds; default `frame_skip=4` advances `0.008`
simulated seconds per Policy step. Forward reward uses the unclipped
step-average x velocity `(x_after-x_before)/seconds_per_step`, not the clipped
next-observation velocity. A walker is healthy only when height and torso angle
are strictly inside both configured open intervals; boundary equality is
unhealthy. Healthy reward is zero on an unhealthy step. By default, an
unhealthy walker terminates; otherwise it can continue until truncation at
1000 steps. The scalar score is mean Episode return.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports forward displacement, x-velocity extrema and backward-step
fraction; torso-height and angle extrema; minimum margins to both health
boundaries; unhealthy and velocity-clipping fractions; action effort;
cumulative forward, control, and survival rewards; exact fall-reason counts;
Policy failures; and bounded trace coverage.

`trace.jsonl` retains at most four Episodes with complete Policy observations,
exact named controls, gear-scaled torques, timing, start/current/running x
positions, clipped-versus-reward velocity, health bounds and margins, reward
decomposition, cumulative return, and explicit terminal reason. Position
metrics remain public when current x is excluded from the observation,
matching the existing Gymnasium `info` contract.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/walker2d
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
