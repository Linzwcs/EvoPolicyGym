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

The spec defines image axes and channels, view orientation, carried-object and
compass encodings, the exact reward formula, and termination rules. Feedback
and `trace.jsonl` expose key/door colors and discovery steps, key pickup/drop,
door opening as an intermediate milestone, target discovery and approach,
current carried and front objects, failed pickup/drop/toggle attempts,
remaining horizon, observation novelty, ineffective Actions, and per-Action
usage. They also identify the irreversible case where `toggle` destroys the
mission box without terminating. No seeds, private identity, or Host paths are
published. The packaged baseline uses public observations only. Long traces
retain the first 128 and final 32 transitions, with retained and omitted
coverage reported explicitly. Each
traced observation also lists visible objects with their color and state.
`ProcessExecution.unsafe()` is used only for trusted baseline tests and is not
a sandbox.
