# EvoPolicyGym Swimmer Benchmark

This independently installable distribution adapts Gymnasium `Swimmer-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from swimmer import SwimmerBenchmark, SwimmerConfig, baseline_program

benchmark = SwimmerBenchmark(
    SwimmerConfig(
        frame_skip=4,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.0001,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    )
)
```

All listed parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `swimmer.xml` model is
fixed. Custom XML paths and rendering/camera settings are Host-owned and never
cross the Policy boundary.

## Contract

The default Policy observation contains eight named floats: front-link and two
rotor angles, front-tip x/y velocity, and the three corresponding angular
velocities. With `exclude_current_positions_from_observation=False`,
`tip_x_position` and `tip_y_position` are prepended.

An Action is an exact two-float list containing both rotor torques, each in
`[-1.0, 1.0]`. Integers, tuples, non-finite values, and out-of-range values are
rejected rather than converted or clipped.

Swimmer never terminates naturally and truncates after 1000 steps. Reward is
weighted forward x velocity minus weighted squared control magnitude. The
scalar Benchmark score is mean Episode return. Gymnasium's published solution
threshold is `360`.

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
cd environments/gymnasium/mujoco/swimmer
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
