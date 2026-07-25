# EvoPolicyGym Reacher Benchmark

This independently installable distribution adapts Gymnasium `Reacher-v5`
through EvoPolicyGym's public authoring SPI and official packaged MuJoCo model.

## Public interface

```python
from reacher import ReacherBenchmark, ReacherConfig, baseline_program

benchmark = ReacherBenchmark(
    ReacherConfig(
        frame_skip=2,
        reward_dist_weight=1.0,
        reward_control_weight=1.0,
    )
)
```

Simulation cadence and reward weights are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. Gymnasium's official `reacher.xml` model is
fixed. Custom XML paths and rendering/camera settings are Host-owned and never
cross the Policy boundary.

## Contract

The Policy receives ten named floats:

- sine and cosine of both arm joint angles;
- target x/y position;
- angular velocity of both joints;
- fingertip-minus-target x/y displacement.

An Action is an exact two-float list containing the first and second joint
torques, each in `[-1.0, 1.0]`. Integers, tuples, non-finite values, and
out-of-range values are rejected rather than converted or clipped.

Reacher does not terminate naturally and truncates after 50 steps. Each reward
is the negative weighted fingertip distance minus the weighted squared action
magnitude. The scalar Benchmark score is mean Episode return; Gymnasium's
published solution threshold is `-3.75`.

Policy failure receives a `-1000` return. The packaged baseline applies zero
torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, final fingertip distance, Policy
failures, and bounded trace coverage. `trace.jsonl` retains at most eight
Episodes with complete observations, exact Actions, rewards, and termination
flags.

Environment seeds, Policy seeds, Host paths, credentials, model paths, and
private runtime evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/mujoco/reacher
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
