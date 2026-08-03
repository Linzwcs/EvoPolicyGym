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

All listed parameters plus cadence, actuator gears, reward formula, and horizon are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `swimmer.xml` model is
fixed. Custom XML paths and rendering/camera settings are Host-owned and never
cross the Policy boundary.

## Contract

The default Policy observation contains eight named `float64`-derived values:
absolute front-link angle, two successive relative rotor angles, instantaneous
front-tip x/y qvel, and the three corresponding angular qvel values. With
`exclude_current_positions_from_observation=False`,
`tip_x_position` and `tip_y_position` are prepended.

An Action is an exact two-float list containing both rotor controls, each in
`[-1.0, 1.0]`. Both official actuators apply gear `150`, so controls are not
direct torques in newton-meters. Feedback reports both values. Integers,
tuples, non-finite values, and out-of-range values are rejected rather than
converted or clipped.

The model timestep is `0.01` seconds; default `frame_skip=4` advances `0.04`
simulated seconds per Policy step. Swimmer never terminates naturally and
truncates after 1000 steps. Reward uses the step-average x velocity computed as
`(x_after - x_before) / seconds_per_step`, which is distinct from the
instantaneous `tip_x_velocity` in the next observation, minus weighted squared
control magnitude. The scalar score is mean Episode return. Gymnasium's
published solution threshold is `360`.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports forward and lateral displacement, net displacement, path
length, lateral-drift maximum, mean/minimum/maximum x velocity, lateral
velocity, backward-step fraction, action effort, cumulative forward and
control rewards, outcomes, Policy failures, and bounded trace coverage.

`trace.jsonl` retains at most four Episodes with complete Policy observations,
exact named controls, gear-scaled torques, timing, start/current/running
position extrema, both instantaneous and reward velocities, current angles,
reward decomposition, cumulative return, and explicit terminal reason. The
position metrics remain public even when current positions are excluded from
the observation, matching the existing public Gymnasium `info` contract.

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
