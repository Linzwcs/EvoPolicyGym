# MiniGrid Unlock Benchmark

An independently installable EvoPolicyGym Benchmark for
[MiniGrid Unlock](https://minigrid.farama.org/environments/minigrid/UnlockEnv/).
The Policy must find the key matching a locked door and open it.

```python
from minigrid_unlock import UnlockBenchmark, baseline_program

benchmark = UnlockBenchmark()
program = baseline_program()
```

The spec defines image axes and channels, view orientation, carried-key and
compass encodings, the exact reward formula, and termination rules. Feedback
and the bounded semantic trace expose key and door discovery colors and first
seen steps, key pickup/drop state, whether the carried key matches the observed
door, the object in front before interaction, successful opening, failed
pickup/drop/toggle attempts, remaining horizon, observation novelty,
ineffective Actions, and per-Action usage without publishing seeds, private
identity, or Host paths. The baseline uses public observations only. Long
traces retain the first 128 and final 32 transitions, with retained and omitted
coverage reported explicitly. Each traced observation also lists visible
objects with their color and state. `ProcessExecution.unsafe()` is used only
for trusted packaged baseline tests and is not a sandbox.
