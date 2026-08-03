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

The primary metric is mean return. Feedback includes bounded per-Episode
summaries and a diagnostic trace covering at most 32 steps in each of four
Episodes. Long Episodes retain their first and last steps, bounded samples of
reward and game-variable-change events, and an even timeline sample. Selected
screen frames, game variables, and optional audio are stored losslessly in
compressed NPZ artifacts; optional notifications remain inline. PNG contact
sheets and bounded animated GIF replays expose the visual trajectory, action
meaning, reward, and primary game variables. Omitted Episodes, steps, and replay
frames are reported explicitly, without seeds, paths, or WAD identity.

Commercial Doom/Doom2 map registrations are intentionally excluded because
their WADs do not ship with ViZDoom. FreeDoom campaign maps can be added as a
separate suite if desired.
