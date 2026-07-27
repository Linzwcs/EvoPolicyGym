# MiniGrid BlockedUnlockPickup Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid BlockedUnlockPickup](https://minigrid.farama.org/environments/minigrid/BlockedUnlockPickupEnv/)
long-horizon manipulation task.

The Policy must move the ball obstructing a locked door, acquire the
matching-color key, unlock the second room, and pick up the mission box.
Feedback and the bounded semantic `trace.jsonl` expose the public progress
ladder without publishing Episode seeds, private identity, or Host paths.

```python
from minigrid_blocked_unlock_pickup import (
    BlockedUnlockPickupBenchmark,
    baseline_program,
)

benchmark = BlockedUnlockPickupBenchmark()
program = baseline_program()
```

The packaged baseline builds a relative map from Policy-visible observations
and implements the complete object-moving and unlocking state machine. Tests
using `ProcessExecution.unsafe()` execute trusted packaged code only; that
backend is not a sandbox.

