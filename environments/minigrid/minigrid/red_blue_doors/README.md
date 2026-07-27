# MiniGrid RedBlueDoors Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid RedBlueDoors](https://minigrid.farama.org/environments/minigrid/RedBlueDoorEnv/)
ordered interaction task.

The Policy receives the upstream egocentric symbolic observation and must open
the red door before opening the blue door. Opening blue first ends the Episode
with zero reward. Available profiles are `6x6` and `8x8`; the Host selects one
before a Run and it contributes to the environment digest.

```python
from minigrid_red_blue_doors import (
    RedBlueDoorsBenchmark,
    RedBlueDoorsConfig,
    baseline_program,
)

benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="8x8"))
program = baseline_program()
```

Feedback and the bounded `trace.jsonl` report door discovery, red-door
completion, order errors, success, returns, steps, truncation, and Policy
failures without publishing seeds, private identity, or Host paths.

The packaged baseline uses only public observations to map the central room
and plans to the red and blue doors in mission order.

Tests using `ProcessExecution.unsafe()` execute trusted packaged code only.
That backend is a local process mechanism, not a sandbox.

