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

The primary metric is mean Episode return. Feedback publishes per-Episode
action/control and speed summaries, crash and success steps, and a diagnostic
trace covering at most 48 steps in each of four Episodes. Long Episodes retain
their first and last steps, bounded crash/success/terminal events, and an even
timeline sample.

Every selected observation is stored losslessly in a compressed NPZ artifact.
The trace references exact decision and result arrays and adds a bounded,
profile-specific semantic view:

- Kinematics profiles label ego and nearby-vehicle features;
- Time-to-Collision profiles report collision-cost and earliest-risk indices;
- Parking reports achieved/desired goals and position, velocity, and heading
  errors;
- Racetrack reports occupancy and on-road grid statistics;
- Lane Keeping reports state, derivative, reference state, and tracking error.

The public Benchmark specification now declares the exact shapes, dtypes,
features, axes, and profile-specific Action meanings. Omitted Episodes and
steps are explicit. Environment seeds and Host identities are never published.
