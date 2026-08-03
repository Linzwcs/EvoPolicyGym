# MiniGrid DistShift Benchmark

An independently installable EvoPolicyGym Benchmark for both MiniGrid
`DistShift` registrations.

```python
from minigrid_dist_shift import DistShiftBenchmark, DistShiftConfig

benchmark = DistShiftBenchmark(DistShiftConfig(profile="shift1"))
```

The Policy must reach the goal without entering lava under the selected
distribution-shift layout. The spec defines image axes
`[view_x, view_y, channel]`, channels `[object, color, state]`, the agent at
`(3,6)`, view orientation, compass and symbolic encodings, the exact
time-decaying reward, and every terminal condition.

Feedback distinguishes seeing lava from entering it, and reports when lava
and the goal were first seen, remaining horizon, observation novelty,
ineffective Actions, per-Action usage, success, timeout, and Policy-failure
outcomes. In particular, it identifies Episodes where the visible goal was
never reached. The bounded semantic trace contains no seeds, private identity,
or Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
