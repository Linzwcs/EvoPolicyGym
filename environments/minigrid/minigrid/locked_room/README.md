# MiniGrid LockedRoom Benchmark

An independently installable EvoPolicyGym Benchmark for
[MiniGrid LockedRoom](https://minigrid.farama.org/environments/minigrid/LockedRoomEnv/).
The Policy searches six rooms, follows the mission to the target-color key,
unlocks the matching room, and reaches its goal.

```python
from minigrid_locked_room import LockedRoomBenchmark, baseline_program
```

The spec defines image axes and channels, view orientation, carried-key and
compass encodings, the exact reward formula, and termination rules. Feedback
and the bounded semantic trace separate the mission's target-key color from
the key-room door color, and report key-room entry, key discovery/acquisition,
target-door discovery/opening, goal discovery and entry, current front object,
remaining horizon, failed interactions, observation novelty, ineffective
Actions, and per-Action usage. Door statistics distinguish unique door colors
opened from repeated open and close events. No seeds, private identity, or
Host paths are published. The baseline uses public observations only. Long
traces retain the first 128 and final 32 transitions, with retained and omitted
coverage reported explicitly. Each traced observation also lists visible
objects with their color and state.
`ProcessExecution.unsafe()` is used only for trusted baseline tests and is not
a sandbox.
