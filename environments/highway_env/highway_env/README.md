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

For the nine profiles whose upstream Environments support `rgb_array`, every
Episode with at least one valid transition also preserves the renderer's
original RGB frames losslessly in
`rendered-frames.npz`, together with step indices, rewards, reward presence,
and cumulative returns. A smaller annotated GIF is derived from the same
frames for convenient inspection; it is not the sole visual evidence. The
Host captures the initial scene, first result, terminal result, and a fixed
stride of intermediate results, with at most 42 frames per Episode. Short
Episodes capture every step. Each manifest states separately whether capture
is complete for that schedule and whether the schedule covers every Episode
step. Raw RGB values are removed from the JSONL trace and never become Policy
observations. Both visual artifacts use `retention="bulk"`.
The upstream `lane-keeping-v0` constructor does not support `render_mode`; its
Episodes therefore publish an explicit unavailable visual-evidence manifest
instead of changing the task or synthesizing a renderer.

The public Benchmark specification now declares the exact shapes, dtypes,
features, axes, and profile-specific Action meanings. Omitted Episodes and
steps are explicit. Environment seeds and Host identities are never published.
