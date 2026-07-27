# MiniGrid Unlock Benchmark

An independently installable EvoPolicyGym Benchmark for
[MiniGrid Unlock](https://minigrid.farama.org/environments/minigrid/UnlockEnv/).
The Policy must find the key matching a locked door and open it.

```python
from minigrid_unlock import UnlockBenchmark, baseline_program

benchmark = UnlockBenchmark()
program = baseline_program()
```

Feedback and the bounded semantic trace expose key discovery, acquisition, and
door-opening progress without publishing seeds, private identity, or Host
paths. The baseline uses public observations only. `ProcessExecution.unsafe()`
is used only for trusted packaged baseline tests and is not a sandbox.

