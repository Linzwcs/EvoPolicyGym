# MiniGrid Crossing Benchmark

An independently installable EvoPolicyGym Benchmark for the MiniGrid
`LavaCrossing` and `SimpleCrossing` registrations. All eight upstream
size/crossing/obstacle combinations are explicit profiles.

```python
from minigrid_crossing import CrossingBenchmark, CrossingConfig

benchmark = CrossingBenchmark(CrossingConfig(profile="lava-S9-N3"))
```

The Policy must safely discover openings through generated rivers and reach
the opposite-corner goal. Feedback reports goal discovery and hazard entry;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.

