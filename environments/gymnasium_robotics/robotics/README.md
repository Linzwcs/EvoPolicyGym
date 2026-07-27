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

The primary metric is success rate. Mean return, termination counts, and a
bounded public transition trace are also published. FrankaKitchen traces expose
only task-completion counts, never Host paths, seeds, or simulator internals.
Multi-agent MaMuJoCo environments intentionally remain outside this
single-Policy ABI.

