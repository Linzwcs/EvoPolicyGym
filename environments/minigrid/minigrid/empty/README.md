# MiniGrid Empty Benchmark

An independently installable EvoPolicyGym Benchmark for all six MiniGrid
`Empty` fixed- and random-start registrations.

```python
from minigrid_empty import EmptyBenchmark, EmptyConfig

benchmark = EmptyBenchmark(EmptyConfig(profile="16x16"))
```

The Policy must navigate an empty room and reach the opposite-corner goal.
Feedback reports goal discovery and success;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
