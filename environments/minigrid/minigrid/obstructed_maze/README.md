# MiniGrid ObstructedMaze Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid ObstructedMaze](https://minigrid.farama.org/environments/minigrid/ObstructedMazeEnv/)
long-horizon manipulation task.

Depending on the selected profile, the Policy must open grey boxes containing
keys, pick up and drop green balls obstructing doors, unlock rooms, and finally
pick up the blue mission ball. The blue target and green blockers are distinct;
only target pickup terminates successfully. `drop` is required to relocate a
blocker and free the carrying slot, while `done` is an unused no-op.

Feedback and the bounded semantic `trace.jsonl` expose discovery steps,
box-opening, blocker pickup/drop, key pickup/drop, locked and unlocked door
events, target discovery, failed interactions, target pickups blocked by full
hands, observation novelty, ineffective actions, action mix, and the exact
terminal reason. They publish no Episode seeds, private identity, or Host
paths.

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

The legacy `2Dlhb-v0`, `1Q-v0`, `2Q-v0`, and `Full-v0` registrations retain an
upstream generation flaw that can place a green blocker over a key and make a
generated task structurally unsolvable. This risk is explicit in Benchmark
metadata and Feedback; use the corresponding `v1` profile for RL datasets that
must avoid it.
