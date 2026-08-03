# MiniGrid Crossing Benchmark

An independently installable EvoPolicyGym Benchmark for the MiniGrid
`LavaCrossing` and `SimpleCrossing` registrations. All eight upstream
size/crossing/obstacle combinations are explicit profiles.

```python
from minigrid_crossing import CrossingBenchmark, CrossingConfig

benchmark = CrossingBenchmark(CrossingConfig(profile="lava-S9-N3"))
```

The Policy must safely discover openings through generated rivers and reach
the opposite-corner goal. The spec defines the image as
`[view_x, view_y, channel]`, with channels `[object, color, state]`, the agent
at `(3,6)`, forward toward decreasing `view_y`, compass codes, every symbolic
code, the exact time-decaying success reward, and all termination conditions.

Feedback distinguishes seeing lava (`hazard_found`) from actually entering it
(`hazard_entered`). It also reports first discovery steps, remaining horizon,
observation novelty, ineffective Actions, per-Action usage, success, timeout,
and Policy-failure outcomes. The bounded semantic trace contains no seeds,
private identity, or Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
