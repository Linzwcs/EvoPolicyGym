# MiniGrid FourRooms Benchmark

An independently installable EvoPolicyGym Benchmark for the MiniGrid
`FourRooms` registration.

```python
from minigrid_four_rooms import FourRoomsBenchmark

benchmark = FourRoomsBenchmark()
```

The Policy must explore generated openings between four rooms and reach the
randomly placed goal. The spec publishes image axis/channel order, egocentric
view orientation, global direction encoding, object/color/state codes, Action
meanings, horizon, and the time-decaying success reward formula.

Feedback reports goal visibility and first-discovery step, success, remaining
horizon, unique observations and novelty rate, repeated/ineffective Actions,
per-Action usage, cumulative return, and terminal reason. It deliberately does
not expose the hidden global position or full maze.

The bounded semantic trace contains no seeds, private identity, or Host paths.
Long traces retain the first 128 and final 32 transitions, with retained and
omitted coverage reported explicitly. Each traced observation also lists
visible objects with their color and state.
The packaged baseline never moves into an unknown or observed obstacle cell.
The fixed public horizon is 256 steps (the upstream default is 100).
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
