# EvoPolicyGym HumanoidStandup Benchmark

This independently installable distribution adapts Gymnasium
`HumanoidStandup-v5` through EvoPolicyGym's public authoring SPI and official
packaged MuJoCo model.

## Public interface

```python
from humanoid_standup import (
    HumanoidStandupBenchmark,
    HumanoidStandupConfig,
    baseline_program,
)

benchmark = HumanoidStandupBenchmark(
    HumanoidStandupConfig(
        frame_skip=5,
        ctrl_cost_weight=0.1,
        impact_cost_weight=0.0000005,
        impact_cost_range=(None, 10.0),
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
unbounded lower impact-cost clamp. Gymnasium's official
`humanoidstandup.xml` model is fixed. Custom XML paths and rendering/camera
settings are Host-owned and never cross the Policy boundary.

Gymnasium 1.3.0 accepts `uph_cost_weight`, but its `HumanoidStandup-v5`
implementation never uses that stored value in reward computation. This
Benchmark intentionally does not expose that no-op argument, so changing a
parameter cannot change identity without changing behavior.

## Contract

The default Policy observation contains 45 named torso and joint state floats
plus four optional semantic objects: 130 body-inertia values, 78 center-of-mass
velocity values, 17 actuator-force values, and 78 external-force values. These
objects preserve all 348 default Gymnasium values while grouping them by body
or joint. Each inclusion flag removes its corresponding object; disabling
position exclusion adds root-qpos torso x/y. The actuator-force object follows
generalized-state order (`abdomen_z` before `abdomen_y`), whereas Action order
begins `abdomen_y, abdomen_z, abdomen_x`.

An Action is an exact 17-float list containing abdomen, hip, knee, shoulder,
and elbow controls, each in `[-0.4, 0.4]`. Official actuator gears are
published for every component: most abdomen/hip axes use `100`, hip-y uses
`300`, knees use `200`, and shoulder/elbow actuators use `25`. Integers,
tuples, non-finite values, and out-of-range values are rejected rather than
converted or clipped.

HumanoidStandup never terminates naturally and truncates after 1000 steps.
Reward is absolute torso height divided by the MuJoCo model timestep, plus
one, minus control and impact costs. This follows the pinned implementation,
including its use of the `0.003`-second model timestep rather than the default
`0.015`-second configured frame duration. The official model's nominal prone
root z is `0.105m`, and reset noise produces initial heights near that value;
the upstream generated prose that describes a `1.4m` initial height does not
match this packaged model. Because the reward uses absolute height rather than
height change, feedback reports both absolute height and gain from reset.
The scalar Benchmark score is mean Episode return.

Policy failure receives a configuration-scaled negative return. The packaged
baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return and steps; initial/final/minimum/maximum torso
height and maximum gain from reset; vertical velocity and upward-step fraction;
quaternion-derived torso tilt; horizontal drift; action use and gear-scaled
controls; actuator, tendon, and external-force diagnostics when their source
observations are enabled; complete current and cumulative reward decomposition;
time-limit outcomes; Policy failures; and bounded trace coverage.
`trace.jsonl` retains at most four Episodes and at most 100 transitions per
retained Episode. Transitions are sampled uniformly with the first and final
transition included. Every retained transition contains complete nested
observations, named Actions, next observations, all diagnostics, and terminal
reason. Episode rows and Feedback publish the real step count, sampling mode,
retained transition count, and omitted transition count. Disabled optional
observations remain unavailable rather than being leaked back through
feedback.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/humanoid_standup
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
