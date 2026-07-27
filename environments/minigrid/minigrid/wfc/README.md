# MiniGrid WFC Benchmark

An independently installable EvoPolicyGym Benchmark family for all 22
MiniGrid Wave Function Collapse presets.

```python
from minigrid_wfc import WFCBenchmark, WFCConfig

benchmark = WFCBenchmark(
    WFCConfig(profile="MazeSimple", size=25)
)
```

The Policy must navigate a generated connected maze and reach its random goal.
Feedback reports goal discovery and success;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
The Host selects a preset profile and size (`15` or `25`) before the Run.
Generation-heavy profiles remain explicit rather than being mixed into an
Episode pool.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
