# MiniGrid ObstructedMaze Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid ObstructedMaze](https://minigrid.farama.org/environments/minigrid/ObstructedMazeEnv/)
long-horizon manipulation task.

The Policy must open boxes containing keys, move balls obstructing locked
doors, unlock the selected maze, and pick up the blue mission ball.
Feedback and the bounded semantic `trace.jsonl` expose the public progress
ladder without publishing Episode seeds, private identity, or Host paths.

```python
from minigrid_obstructed_maze import (
    ObstructedMazeBenchmark,
    ObstructedMazeConfig,
    baseline_program,
)

benchmark = ObstructedMazeBenchmark(
    ObstructedMazeConfig(profile="Full-v1")
)
program = baseline_program()
```

The packaged baseline builds a relative map from Policy-visible observations
and implements the complete object-moving and unlocking state machine. Tests
using `ProcessExecution.unsafe()` execute trusted packaged code only; that
backend is not a sandbox.

All 13 upstream registrations are profiles, from `1Dl-v0` through `Full-v1`.
The Host selects one profile before the Run; the profile is visible to the
Agent, fixed for the Run, and included in the environment digest.
