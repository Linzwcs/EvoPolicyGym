# EvoPolicyGym Ant Benchmark

This independently installable distribution adapts Gymnasium `Ant-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from ant import AntBenchmark, AntConfig, baseline_program

benchmark = AntBenchmark(
    AntConfig(
        frame_skip=5,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.5,
        contact_cost_weight=0.0005,
        healthy_reward=1.0,
        main_body=1,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        include_cfrc_ext_in_observation=True,
    )
)
```

All listed parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. `main_body` is restricted to non-world
body indices `1..13` in Gymnasium's official `ant.xml`. That model is fixed;
custom XML paths and rendering/camera settings are Host-owned and never cross
the Policy boundary.

## Contract

The default Policy observation contains 27 named torso and joint state floats,
plus a nested `contact_forces` object containing all 78 clipped external force
and torque values for the 13 non-world bodies. Setting
`include_cfrc_ext_in_observation=False` omits that object. Setting
`exclude_current_positions_from_observation=False` additionally exposes
`torso_x_position` and `torso_y_position`.

An Action is an exact eight-float list containing hip and ankle controls for
the four legs, each in `[-1.0, 1.0]`. The actuator order is back-right,
front-left, front-right, back-left, with hip then ankle for each pair; this is
the model actuator order `hip_4, ankle_4, hip_1, ...`, not qpos joint order.
Every actuator has gear `150`, which is published in the spec and exposed as a
gear-scaled control diagnostic. Integers, tuples, non-finite values, and
out-of-range values are rejected rather than converted or clipped.

Reward is exactly:

```text
forward_weight*x_velocity
+ healthy_reward if torso_z is healthy
- control_weight*sum(action²)
- contact_weight*sum(clipped_contact_force_components²)
```

The configured `main_body` supplies forward velocity; its index and semantic
body name are public. Contact values are clipped to `contact_force_range`
before both observation and cost. Health requires a finite state and torso z
inside the inclusive configured range. By default, an unhealthy Ant terminates;
otherwise the Episode truncates after 1000 steps. MuJoCo's model timestep is
`0.01 s`, so one Policy step spans `0.01 × frame_skip` seconds.
The scalar Benchmark score is mean Episode return. Gymnasium's published
solution threshold is `6000`.

Policy failure receives a configuration-scaled negative return. The packaged
baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports unhealthy versus time-limit outcomes, final and trajectory x
positions, distance from origin, mean forward velocity, healthy-step fraction,
torso tilt, minimum health-height margin, action magnitude, contact intensity,
all four cumulative reward terms, Policy failures, and bounded trace coverage.

`trace.jsonl` retains at most one Episode with complete nested Policy
observations and exact Actions. Per-step metrics contain joint-named controls,
gear-scaled controls, elapsed time, horizontal kinematics, torso height and
tilt, quaternion error, health margins, clipped-contact summaries, current and
cumulative reward decomposition, trajectory extrema, and terminal reason.
Episode rows publish the corresponding aggregate diagnostics and outcome.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/ant
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
