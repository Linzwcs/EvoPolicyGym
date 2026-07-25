# EvoPolicyGym Hopper Benchmark

This independently installable distribution adapts Gymnasium `Hopper-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from hopper import HopperBenchmark, HopperConfig, baseline_program

benchmark = HopperBenchmark(
    HopperConfig(
        frame_skip=4,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.001,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, None),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=0.005,
        exclude_current_positions_from_observation=True,
    )
)
```

All listed parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. `None` is the public representation of an
unbounded upper z limit. Gymnasium's official `hopper.xml` model is fixed.
Custom XML paths and rendering/camera settings are Host-owned and never cross
the Policy boundary.

## Contract

The default Policy observation contains eleven named floats: torso height and
angle, three joint angles, torso x/z velocity, and four angular velocities.
With `exclude_current_positions_from_observation=False`,
`torso_x_position` is prepended.

An Action is an exact three-float list containing thigh, leg, and foot rotor
torques, each in `[-1.0, 1.0]`. Integers, tuples, non-finite values, and
out-of-range values are rejected rather than converted or clipped.

Reward combines weighted forward x velocity and the healthy reward, then
subtracts weighted squared control magnitude. By default, an unhealthy Hopper
terminates; otherwise the Episode truncates after 1000 steps. The scalar
Benchmark score is mean Episode return. Gymnasium's published solution
threshold is `3800`.

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
cd environments/gymnasium/mujoco/hopper
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
