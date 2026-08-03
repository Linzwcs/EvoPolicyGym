# MiniGrid RedBlueDoors Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid RedBlueDoors](https://minigrid.farama.org/environments/minigrid/RedBlueDoorEnv/)
ordered interaction task.

The Policy receives the upstream egocentric symbolic observation and must open
the red door before opening the blue door. The red door must still be open
immediately before the blue-door Action: opening blue first, or opening red,
closing it again, and then opening blue, ends the Episode with zero reward.
Available profiles are `6x6` and `8x8`; the Host selects one before a Run and
it contributes to the environment digest.

```python
from minigrid_red_blue_doors import (
    RedBlueDoorsBenchmark,
    RedBlueDoorsConfig,
    baseline_program,
)

benchmark = RedBlueDoorsBenchmark(RedBlueDoorsConfig(profile="8x8"))
program = baseline_program()
```

The spec defines image axes and channels, view orientation, compass and
symbolic encodings, exact reward, and terminal conditions. Feedback exposes
the full ordered funnel: first seeing each door, opening red, reclosing red,
opening blue, and succeeding. It distinguishes blue-before-red from
blue-after-red-was-reclosed, reports the current task stage, remaining horizon,
observation novelty, ineffective Actions, per-Action usage, truncation, and
Policy failures. The bounded `trace.jsonl` publishes the same semantic evidence
without seeds, private identity, or Host paths.

The packaged baseline uses only public observations to map the central room
and plans to the red and blue doors in mission order.

Tests using `ProcessExecution.unsafe()` execute trusted packaged code only.
That backend is a local process mechanism, not a sandbox.
