# MiniGrid LavaGap Benchmark

An independently installable EvoPolicyGym Benchmark for all three MiniGrid
`LavaGap` size registrations.

```python
from minigrid_lava_gap import LavaGapBenchmark, LavaGapConfig

benchmark = LavaGapBenchmark(LavaGapConfig(profile="S7"))
```

The Policy must safely discover the opening through a generated lava strip and
reach the opposite-corner goal. The spec defines image axes
`[view_x, view_y, channel]`, channels `[object, color, state]`, the agent at
`(3,6)`, view orientation, compass and symbolic encodings, the exact
time-decaying reward, and every terminal condition.

Feedback distinguishes seeing lava from entering it, and reports first-seen
steps, remaining horizon, observation novelty, ineffective Actions,
per-Action usage, success, timeout, and Policy-failure outcomes. It also
identifies Episodes where the goal entered the view but was never reached. The
bounded semantic trace contains no seeds, private identity, or Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
