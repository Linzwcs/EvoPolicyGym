# Stable-Retro Airstriker Benchmark

Stable-Retro 1.0.1 provides metadata for 1,030 game integrations, but only
Airstriker includes a redistributable ROM in the wheel. This distribution
therefore exposes exactly `Airstriker-Genesis-v0`; it neither accepts Host ROM
paths nor claims that proprietary games are installed.

```python
from airstriker import AirstrikerBenchmark

benchmark = AirstrikerBenchmark()
```

The Policy receives 224 × 320 RGB frames and returns one of Stable-Retro's 126
restricted discrete controller actions. The Episode starts from the bundled
Level 1 state, ends on the upstream game-over condition, and has a public
18,000-frame fallback horizon. Mean score delta is the primary metric.

Feedback includes bounded per-Episode summaries, explicit controller-button
meanings for used actions, and a diagnostic trace covering at most 32 steps in
each of four Episodes. Long Episodes retain their first and last steps, a
bounded sample of non-zero reward events, and an even sample of the remaining
timeline. The trace references lossless RGB arrays in compressed NPZ artifacts;
PNG contact sheets provide lightweight previews for traced Episodes, and every
Episode publishes its own bounded animated GIF replay. Omitted trace Episodes
and steps plus sampled replay frames are reported explicitly; the Kernel-owned
`feedback.json` retains every Episode result.
