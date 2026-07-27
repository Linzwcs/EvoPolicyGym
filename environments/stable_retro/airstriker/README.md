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
