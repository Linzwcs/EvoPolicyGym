# HighwayEnv Benchmark

This independently installable distribution exposes the ten canonical
single-agent HighwayEnv tasks through EvoPolicyGym's public Benchmark authoring
surface.

The Host selects one fixed `HighwayConfig.profile` for a Run. The selected
profile is visible in the public environment parameters and contributes to the
environment digest. An Agent cannot change it from its Policy.

Available profiles are `highway`, `merge`, `roundabout`, `intersection`,
`two-way`, `exit`, `u-turn`, `parking`, `racetrack`, and `lane-keeping`.
The first seven use strict discrete meta-actions. The final three use strict
continuous actions.

```python
from highway_benchmarks import HighwayBenchmark, HighwayConfig

benchmark = HighwayBenchmark(HighwayConfig(profile="roundabout"))
```

The primary metric is mean Episode return. Feedback also publishes a bounded
transition trace containing public actions, rewards, termination flags, and
public simulator metrics. Environment seeds and Host identities are never
published.

