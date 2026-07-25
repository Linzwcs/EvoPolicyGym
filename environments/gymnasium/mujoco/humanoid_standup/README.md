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
position exclusion adds torso x/y.

An Action is an exact 17-float list containing abdomen, hip, knee, shoulder,
and elbow torques, each in `[-0.4, 0.4]`. Integers, tuples, non-finite values,
and out-of-range values are rejected rather than converted or clipped.

HumanoidStandup never terminates naturally and truncates after 1000 steps.
Reward is absolute torso height divided by the MuJoCo model timestep, plus
one, minus control and impact costs. This follows the pinned implementation,
including its use of model timestep rather than configured frame duration.
The scalar Benchmark score is mean Episode return.

Policy failure receives a configuration-scaled negative return. The packaged
baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, final torso height, Policy failures,
and bounded trace coverage. `trace.jsonl` retains at most four Episodes with
complete nested Policy observations, exact Actions, tendon state, public
positions, reward terms, and termination flags.

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
