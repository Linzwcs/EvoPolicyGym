# MiniGrid LockedRoom Benchmark

An independently installable EvoPolicyGym Benchmark for
[MiniGrid LockedRoom](https://minigrid.farama.org/environments/minigrid/LockedRoomEnv/).
The Policy searches six rooms, follows the mission to the target-color key,
unlocks the matching room, and reaches its goal.

```python
from minigrid_locked_room import LockedRoomBenchmark, baseline_program
```

Feedback and the bounded semantic trace report ordinary doors opened, key
discovery/acquisition, target-door completion, and goal discovery without
publishing seeds, private identity, or Host paths. The baseline uses public
observations only. `ProcessExecution.unsafe()` is used only for trusted
baseline tests and is not a sandbox.

