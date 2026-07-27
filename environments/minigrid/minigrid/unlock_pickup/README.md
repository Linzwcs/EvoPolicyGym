# MiniGrid UnlockPickup Benchmark

An independently installable EvoPolicyGym Benchmark for
[MiniGrid UnlockPickup](https://minigrid.farama.org/environments/minigrid/UnlockPickupEnv/).
The Policy must acquire the matching key, unlock the second room, release the
key, and collect the mission box.

```python
from minigrid_unlock_pickup import UnlockPickupBenchmark, baseline_program

benchmark = UnlockPickupBenchmark()
program = baseline_program()
```

Feedback and `trace.jsonl` expose the public progress ladder without
publishing seeds, private identity, or Host paths. The packaged baseline uses
public observations only. `ProcessExecution.unsafe()` is used only for trusted
baseline tests and is not a sandbox.

