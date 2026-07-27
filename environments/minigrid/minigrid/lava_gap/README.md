# MiniGrid LavaGap Benchmark

An independently installable EvoPolicyGym Benchmark for all three MiniGrid
`LavaGap` size registrations.

```python
from minigrid_lava_gap import LavaGapBenchmark, LavaGapConfig

benchmark = LavaGapBenchmark(LavaGapConfig(profile="S7"))
```

The Policy must safely discover the opening through a generated lava strip and reach
the opposite-corner goal. Feedback reports goal discovery and hazard entry;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
