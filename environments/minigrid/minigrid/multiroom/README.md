# MiniGrid MultiRoom Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid MultiRoom](https://minigrid.farama.org/environments/minigrid/MultiRoomEnv/)
navigation task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and fixed mission. It must discover a chain of rooms, open
the connecting doors, and reach the green goal in the final room. The primary
score is Episode success rate.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/multiroom --extra dev
uv run --project environments/minigrid/minigrid/multiroom \
  python -m unittest discover \
  -s environments/minigrid/minigrid/multiroom/tests
uv build environments/minigrid/minigrid/multiroom
```

## Public API

```python
from minigrid_multiroom import (
    MultiRoomBenchmark,
    MultiRoomConfig,
    baseline_program,
)

benchmark = MultiRoomBenchmark(
    MultiRoomConfig(profile="N6-S10"),
)
program = baseline_program()
```

Available profiles are `N2-S4`, `N4-S5`, `N4-S5-v0-legacy-N6`, and
`N6-S10`. The legacy profile preserves the upstream registration whose name
says four rooms but whose configuration actually generates six. A profile is
Host-selected before a Run and contributes to the environment digest.

Feedback reports success, final-goal discovery, and opened-door counts,
together with return, step, truncation, and Policy-failure statistics.
`trace.jsonl` contains a bounded semantic trace for at most four Episodes,
retaining the first 128 and last 32 transitions of long Episodes. It contains
no Episode seed, private Case identity, or Host path.

The packaged baseline builds a relative map from public egocentric
observations, opens reachable doors, and shortest-path plans to the goal. It
consumes no private environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.

