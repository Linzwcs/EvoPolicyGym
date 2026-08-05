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

The default Policy observation contains seventeen named floats: torso root
height and pitch, six back/front leg joint angles, torso root x/z velocity, and
seven angular velocities. Earlier `front_tip_*` labels were incorrect for the
official model's `rootx`, `rootz`, and `rooty` qpos/qvel entries and have been
replaced with `torso_*`. With
`exclude_current_positions_from_observation=False`, `torso_x_position` is
prepended.

An Action is an exact six-float list containing the back and front thigh, shin,
and foot controls, each in `[-1.0, 1.0]`. The actuator order is back
thigh/shin/foot then front thigh/shin/foot, with respective gears
`[120, 90, 60, 120, 60, 30]`. Integers, tuples, non-finite values, and
out-of-range values are rejected rather than converted or clipped.

HalfCheetah never terminates naturally and truncates after 1000 steps. Reward
is exactly `forward_weight*x_velocity - control_weight*sum(action²)`. MuJoCo's
model timestep is `0.01 s`; one Policy step spans `0.01 × frame_skip` seconds.
The scalar Benchmark score is mean Episode return. Gymnasium's published
solution threshold is `4800`.

Policy failure receives a configuration-scaled return no greater than `-1000`.
The packaged baseline applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports final and net x displacement, trajectory x extrema, mean and
peak forward/backward velocity, forward-step fraction, torso height and pitch
range, action magnitude, cumulative forward reward and control penalty,
time-limit outcomes, Policy failures, and bounded trace coverage.

`trace.jsonl` retains at most four Episodes with complete Policy observations
and exact Actions. Per-step metrics contain joint-named controls, gear-scaled
controls, elapsed time, initial/current/net x position, velocity and direction
fractions, torso pose, current and cumulative reward decomposition, and
terminal reason. Episode rows publish corresponding aggregate diagnostics.

Every Episode with at least one valid transition also preserves every captured
256 × 256 `rgb_array` frame losslessly in `rendered-frames.npz`, together with
step indices, rewards, reward presence, and cumulative returns. The Benchmark
captures the initial pose, first result, terminal result, and a fixed stride of
intermediate results, with at most 42 frames per Episode. An H.264 MP4 at five
frames per second is derived from exactly that sequence for direct playback;
the MP4 is presentation, while the NPZ is the pixel-exact evidence. Both use
`retention="bulk"`, and the manifest states whether capture is complete for
the sampling schedule and whether the schedule covers every Episode step. Raw
RGB tensors are removed from `trace.jsonl` and never become Policy
observations. Rendering failure is diagnostic and does not change physics or
score; zero-step failures publish an explicit unavailable manifest.

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
