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

An Action is an exact eight-float list containing hip and ankle torques for
the four legs, each in `[-1.0, 1.0]`. Integers, tuples, non-finite values, and
out-of-range values are rejected rather than converted or clipped.

Reward combines weighted forward velocity and the healthy reward, then
subtracts weighted squared control and clipped-contact costs. By default, an
unhealthy Ant terminates; otherwise the Episode truncates after 1000 steps.
The scalar Benchmark score is mean Episode return. Gymnasium's published
solution threshold is `6000`.

Policy failure receives a configuration-scaled negative return. The packaged
baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, final x position, Policy failures,
and bounded trace coverage. `trace.jsonl` retains at most four Episodes with
complete nested Policy observations, exact Actions, public kinematic metrics,
reward terms, and termination flags.

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
