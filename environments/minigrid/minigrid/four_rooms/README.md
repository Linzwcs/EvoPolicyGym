# MiniGrid FourRooms Benchmark

An independently installable EvoPolicyGym Benchmark for the MiniGrid
`FourRooms` registration.

```python
from minigrid_four_rooms import FourRoomsBenchmark

benchmark = FourRoomsBenchmark()
```

The Policy must explore generated openings between four rooms and reach the
randomly placed goal. Feedback reports goal discovery and success;
the bounded semantic trace contains no seeds, private identity, or Host paths.
The packaged baseline never moves into an unknown or observed obstacle cell.
The fixed public horizon is 256 steps (the upstream default is 100).
Trusted baseline tests use `ProcessExecution.unsafe()`, which is not a sandbox.
