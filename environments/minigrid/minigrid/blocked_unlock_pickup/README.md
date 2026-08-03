# MiniGrid BlockedUnlockPickup Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid BlockedUnlockPickup](https://minigrid.farama.org/environments/minigrid/BlockedUnlockPickupEnv/)
long-horizon manipulation task.

The Policy must move the ball obstructing a locked door, acquire the
matching-color key, unlock the second room, and pick up the mission box.
The spec defines image axes and channels, view orientation, carried-object and
compass encodings, the exact reward formula, and termination rules. Feedback
and the bounded semantic `trace.jsonl` expose the Policy-visible progress
ladder: first discovery of the blocker, key, door, and target; blocker pickup
and drop; key pickup and drop; door opening; current carried and front objects;
failed pickup/drop/toggle attempts; remaining horizon; observation novelty;
ineffective Actions; and per-Action usage. It also reports the irreversible
case where `toggle` is applied to the mission box: MiniGrid destroys the box
without terminating the Episode. No Episode seed, private identity, or Host
path is published.

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
