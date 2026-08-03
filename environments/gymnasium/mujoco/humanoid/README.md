# EvoPolicyGym Humanoid Benchmark

This independently installable distribution adapts Gymnasium `Humanoid-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from humanoid import HumanoidBenchmark, HumanoidConfig, baseline_program

benchmark = HumanoidBenchmark(
    HumanoidConfig(
        frame_skip=5,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        contact_cost_weight=0.0000005,
        contact_cost_range=(None, 10.0),
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.01,
        exclude_current_positions_from_observation=True,
        include_cinert_in_observation=True,
        include_cvel_in_observation=True,
        include_qfrc_actuator_in_observation=True,
        include_cfrc_ext_in_observation=True,
    )
)
```

All listed parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. `None` is the public representation of an
unbounded lower contact-cost clamp. Gymnasium's official `humanoid.xml` model
is fixed. Custom XML paths and rendering/camera settings are Host-owned and
never cross the Policy boundary.

## Contract

The default Policy observation contains 45 named torso and joint state floats
plus four optional semantic objects: 130 body-inertia values, 78 center-of-mass
velocity values, 17 actuator-force values, and 78 external-force values. These
objects preserve all 348 default Gymnasium values while grouping them by body
or joint. Each inclusion flag removes its corresponding object; disabling
position exclusion adds root-qpos torso x/y. The scalar torso quaternion is in
`w,x,y,z` order. The actuator-force object follows generalized-state order,
whose first entries are `abdomen_z, abdomen_y, abdomen_x`; this intentionally
differs from Action order.

An Action is an exact 17-float list containing abdomen, hip, knee, shoulder,
and elbow controls, each in `[-0.4, 0.4]`. Action order begins
`abdomen_y, abdomen_z, abdomen_x`. Official actuator gears are published for
every component: most abdomen/hip axes use `100`, hip-y uses `300`, knees use
`200`, and all shoulder/elbow actuators use `25`. Integers, tuples, non-finite
values, and out-of-range values are rejected rather than converted or clipped.

Reward combines whole-body center-of-mass x velocity and the healthy reward,
then subtracts squared-control and clamped external-contact costs. The
`x_position`/`y_position` diagnostics are root qpos, while forward reward uses
whole-body center-of-mass displacement; feedback labels this distinction. The
model timestep is `0.003` seconds and default frame skip `5` makes each Policy
step `0.015` simulated seconds. Health requires root torso z to be strictly
inside the configured open interval; an unhealthy step receives no healthy
reward. By default it terminates; otherwise the Episode truncates after 1000
steps. The scalar Benchmark score is mean Episode return.

Policy failure receives a configuration-scaled negative return. The packaged
baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return and steps; root x/y displacement and extrema;
center-of-mass velocity, speed, and forward-step fraction; torso height,
quaternion-derived tilt, health margins and health fraction; action use and
gear-scaled controls; actuator, tendon, and external-force diagnostics when
their source observations are enabled; full reward decomposition; explicit
outcome counts; Policy failures; and bounded trace coverage. `trace.jsonl`
retains at most four Episodes and at most 100 transitions per retained Episode.
Long Episodes are sampled uniformly with the first and final transition
included; shorter Episodes remain complete. Every retained transition contains
complete nested observations, named Actions, current and cumulative
diagnostics, and terminal reason. Episode rows and Feedback publish the real
step count, sampling mode, retained transition count, and omitted transition
count. Disabled optional observations stay unavailable in diagnostics rather
than being leaked back to the Policy through feedback.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/humanoid
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
