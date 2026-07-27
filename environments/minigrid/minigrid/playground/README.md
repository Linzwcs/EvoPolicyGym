# MiniGrid Playground Benchmark

An independently installable EvoPolicyGym Benchmark for the MiniGrid
`Playground` registration.

```python
from minigrid_playground import PlaygroundBenchmark

benchmark = PlaygroundBenchmark()
```

The Policy must open generated doors and visit all nine rooms. Feedback reports
mean room coverage and full-coverage success;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
The fixed public horizon is 1000 steps (the upstream default is 100).
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
