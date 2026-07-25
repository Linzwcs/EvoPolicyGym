# EvoPolicyGym HalfCheetah Benchmark

This independently installable distribution adapts Gymnasium
`HalfCheetah-v5` through EvoPolicyGym's public authoring SPI and official
packaged MuJoCo model.

## Public interface

```python
from half_cheetah import (
    HalfCheetahBenchmark,
    HalfCheetahConfig,
    baseline_program,
)

benchmark = HalfCheetahBenchmark(
    HalfCheetahConfig(
        frame_skip=5,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    )
)
```

All listed parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `half_cheetah.xml`
model is fixed. Custom XML paths and rendering/camera settings are Host-owned
and never cross the Policy boundary.

## Contract

The default Policy observation contains seventeen named floats: front-tip
height and angle, six back/front leg joint angles, front-tip x/z velocity, and
seven angular velocities. With
`exclude_current_positions_from_observation=False`, `front_tip_x_position` is
prepended.

An Action is an exact six-float list containing the back and front thigh, shin,
and foot torques, each in `[-1.0, 1.0]`. Integers, tuples, non-finite values,
and out-of-range values are rejected rather than converted or clipped.

HalfCheetah never terminates naturally and truncates after 1000 steps. Reward
is weighted forward x velocity minus weighted squared control magnitude. The
scalar Benchmark score is mean Episode return. Gymnasium's published solution
threshold is `4800`.

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
cd environments/gymnasium/mujoco/half_cheetah
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
