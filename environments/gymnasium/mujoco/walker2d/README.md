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

All listed parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official
`walker2d_v5.xml` model is fixed. Custom XML paths and rendering/camera
settings are Host-owned and never cross the Policy boundary.

## Contract

The default Policy observation contains seventeen named floats: torso height
and angle, six right/left leg joint angles, torso x/z velocity, and seven
angular velocities. With
`exclude_current_positions_from_observation=False`, `torso_x_position` is
prepended.

An Action is an exact six-float list containing the right and left thigh, leg,
and foot torques, each in `[-1.0, 1.0]`. Integers, tuples, non-finite values,
and out-of-range values are rejected rather than converted or clipped.

Reward combines weighted forward x velocity and the healthy reward, then
subtracts weighted squared control magnitude. By default, an unhealthy walker
terminates; otherwise the Episode truncates after 1000 steps. The scalar
Benchmark score is mean Episode return.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, final x position, Policy failures,
and bounded trace coverage. `trace.jsonl` retains at most four Episodes with
complete Policy observations, exact Actions, public kinematic metrics, reward
terms, and termination flags.

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
