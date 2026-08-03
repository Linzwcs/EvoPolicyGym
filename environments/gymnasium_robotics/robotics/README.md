# Gymnasium-Robotics Benchmark

This independent distribution exposes the currently supported single-agent
Gymnasium-Robotics tasks through EvoPolicyGym's public authoring API.

One Host-selected `RoboticsConfig.profile` is fixed for an entire Run and is
included in the public environment parameters and environment digest. The
profile cannot be changed by a Policy.

The package contains 21 profiles across Fetch, Point/Ant Maze, Adroit Hand,
Shadow Dexterous Hand (including boolean and continuous touch-sensor variants),
and FrankaKitchen. `ROBOTICS_PROFILES` is the canonical profile list.

```python
from robotics_benchmarks import RoboticsBenchmark, RoboticsConfig

benchmark = RoboticsBenchmark(
    RoboticsConfig(profile="fetch-pick-and-place")
)
```

The primary metric is success rate: an Episode counts as solved if the upstream
success condition is reached on any step. Most profiles continue after success,
so current success, first success, successful-step fraction, and later loss of
the goal are reported separately. Goal-conditioned profiles expose current,
initial, and best goal errors plus per-step improvement. Adroit profiles retain
their upstream dense shaping and add public-state task progress where the
observation supports it. Control diagnostics report action magnitude,
saturation, zero-action rate, and state motion. FrankaKitchen reports completed,
newly completed, and remaining public task names as well as completion fraction.

Each traced transition contains the Policy-visible observation, Action, reward,
next observation, and public metrics. Episodes longer than 160 steps retain the
first 128 and final 32 steps, with retained and omitted counts reported
explicitly. No Host paths, seeds, simulator objects, or other private identity
enter Feedback.
Multi-agent MaMuJoCo environments intentionally remain outside this
single-Policy ABI.
