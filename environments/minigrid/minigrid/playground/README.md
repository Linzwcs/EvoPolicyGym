# MiniGrid Playground Benchmark

An independently installable EvoPolicyGym Benchmark for the MiniGrid
`Playground` registration.

```python
from minigrid_playground import PlaygroundBenchmark

benchmark = PlaygroundBenchmark()
```

The upstream Playground has no goal, reward, or natural termination. This
Benchmark defines a coverage task: the starting room counts toward coverage,
each first entry into another geometric room rewards `1`, and entering all nine
rooms terminates successfully. The maximum return is therefore `8`.

The Policy must open generated doors and visit all nine rooms. It may also need
to pick up and drop portable objects that obstruct exploration; `done` is an
unused no-op. Feedback reports the first-entry step for coverage levels 2–9,
door and doorway events, object movement, failed interactions, blocked moves,
steps since the last new room, observation novelty, ineffective actions, action
mix, exact terminal reason, mean room coverage, and full-coverage success. The
bounded semantic trace contains no seeds, private room coordinate, identity, or
Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
The fixed public horizon is 1000 steps (the upstream default is 100).
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
