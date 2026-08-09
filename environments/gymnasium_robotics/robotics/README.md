# Gymnasium-Robotics Benchmark

This independent distribution exposes the currently supported single-agent
Gymnasium-Robotics tasks through EvoPolicyGym's public authoring API.

One Host-selected `RoboticsConfig.profile` is fixed for an entire Run and is
included in the public environment parameters and environment digest. The
profile cannot be changed by a Policy.

Policy-visible numeric observations use the bounded `TensorValue` ABI rather
than iterable Python arrays. The public environment parameters give the exact
little-endian `float64` decoding pattern used by Policy authors.

The package contains 21 profiles across Fetch, Point/Ant Maze, Adroit Hand,
Shadow Dexterous Hand (including boolean and continuous touch-sensor variants),
and FrankaKitchen. `ROBOTICS_PROFILES` is the canonical profile list.

```python
from robotics_benchmarks import RoboticsBenchmark, RoboticsConfig

benchmark = RoboticsBenchmark(
    RoboticsConfig(profile="fetch-pick-and-place")
)
```

The primary metric is mean Episode return from the upstream environment.
Success rate remains separately reported: an Episode counts as solved if the
upstream success condition is reached on any step. A Policy failure receives a
profile-specific return below a valid unsuccessful Episode. Most profiles
continue after success, so current success, first success, successful-step
fraction, and later loss of the goal are reported separately. Goal-conditioned profiles expose current,
initial, and best goal errors plus per-step improvement. Adroit profiles retain
their upstream dense shaping and add public-state task progress where the
observation supports it. Control diagnostics report action magnitude,
saturation, zero-action rate, and state motion. FrankaKitchen reports completed,
newly completed, and remaining public task names as well as completion fraction.

Each traced transition contains the Policy-visible observation, Action, reward,
next observation, and public metrics. Episodes longer than 160 steps retain the
first 128 and final 32 steps, with retained and omitted counts reported
explicitly. No Host paths, seeds, simulator objects, or other private identity
enter Feedback. In addition to that bounded JSONL trace, every Episode with at
least one valid transition preserves every captured 128x128 camera frame
losslessly in `rendered-frames.npz`, together with step indices, rewards,
reward presence, and cumulative returns. A bounded animated GIF is derived
from the same frames for convenient inspection; it is not the sole visual
evidence. Fetch, AntMaze, and Adroit use stable named cameras. FrankaKitchen
uses the versioned `franka-kitchen-overview-v1` free-camera pose so the robot,
burners, sink, microwave, cabinets, and kettle remain visible in one fixed
view. PointMaze and Shadow Hand use the simulator's free camera. Exact feedback
camera parameters are part of the public Environment parameters and therefore
the Environment digest.

The Host captures the initial state, first result, terminal result, and an
adaptive stride of intermediate results, with at most 42 frames per Episode.
Raw RGB values exist in Host-side Step metrics during evaluation, are removed
from JSONL traces, and never become Policy observations. The lossless frame
evidence and derived GIF use `retention="bulk"`. Each manifest states whether
capture is complete for the configured sampling schedule and whether it covers
every Episode step.

Feedback includes one visual-evidence manifest for every Episode. A zero-step Policy
failure has no public post-reset artifact channel, so it is retained as an
explicit unavailable visual-evidence result rather than being silently omitted.

Multi-agent MaMuJoCo environments intentionally remain outside this
single-Policy ABI.
