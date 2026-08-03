# MiniGrid Empty Benchmark

An independently installable EvoPolicyGym Benchmark for all six MiniGrid
`Empty` fixed- and random-start registrations.

```python
from minigrid_empty import EmptyBenchmark, EmptyConfig

benchmark = EmptyBenchmark(EmptyConfig(profile="16x16"))
```

The Policy must navigate an empty room and reach the opposite-corner goal.
The observation is an agent-centric `7x7x3` tensor with axes
`[view_x, view_y, channel]` and channels `[object, color, state]`. The agent is
at view coordinate `(3,6)`, forward decreases `view_y`, and right increases
`view_x`. Global direction codes are `0=east`, `1=south`, `2=west`, and
`3=north`. These conventions, every object/color/state code, all seven Action
meanings, and the time-decaying success reward
`1 - 0.9*step_count/max_episode_steps` are published in the Benchmark spec.

Feedback reports goal visibility and first-discovery step, success, observation
novelty, repeated/ineffective Action fraction, per-Action use, remaining
horizon, cumulative return, and terminal reason. This does not expose the
hidden global agent position or full grid.

The bounded semantic trace contains no seeds, private identity, or Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
