# MiniGrid DistShift Benchmark

An independently installable EvoPolicyGym Benchmark for both MiniGrid
`DistShift` registrations.

```python
from minigrid_dist_shift import DistShiftBenchmark, DistShiftConfig

benchmark = DistShiftBenchmark(DistShiftConfig(profile="shift1"))
```

The Policy must reach the goal without entering lava under the selected
distribution-shift layout. Feedback reports goal discovery and hazard entry;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
