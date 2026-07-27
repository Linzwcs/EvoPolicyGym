# ViZDoom Benchmark

This independent distribution exposes the 12 standard ViZDoom 1.3.0 scenarios
whose WAD/config assets ship in the upstream wheel. The Host fixes one
`ViZDoomConfig.profile` per Run.

```python
from vizdoom_benchmarks import ViZDoomBenchmark, ViZDoomConfig

benchmark = ViZDoomBenchmark(ViZDoomConfig(profile="deadly-corridor"))
```

RGB frames, game variables, optional audio, and notifications cross the Policy
boundary only as bounded values and canonical tensors. Most profiles use a
strict discrete action. Deathmatch uses a strict object containing one discrete
binary-button selection and three continuous delta controls.

The primary metric is mean return. Feedback includes a bounded transition trace
without frames, seeds, paths, or WAD identity. Commercial Doom/Doom2 map
registrations are intentionally excluded because their WADs do not ship with
ViZDoom. FreeDoom campaign maps can be added as a separate suite if desired.
