# MiniGrid DynamicObstacles Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid DynamicObstacles](https://minigrid.farama.org/environments/minigrid/DynamicObstaclesEnv/)
navigation task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and fixed mission. It must reach the green goal while grey
balls move locally before every action. The only valid actions are left,
right, and forward; collisions end the Episode with reward `-1`. The primary
score is collision-free Episode success rate.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/dynamic_obstacles --extra dev
uv run --project environments/minigrid/minigrid/dynamic_obstacles \
  python -m unittest discover \
  -s environments/minigrid/minigrid/dynamic_obstacles/tests
uv build environments/minigrid/minigrid/dynamic_obstacles
```

## Public API

```python
from minigrid_dynamic_obstacles import (
    DynamicObstaclesBenchmark,
    DynamicObstaclesConfig,
    baseline_program,
)

benchmark = DynamicObstaclesBenchmark(
    DynamicObstaclesConfig(profile="8x8-N4"),
)
program = baseline_program()
```

Available profiles are `5x5-N2`, `5x5-N2-random`, `6x6-N3`,
`6x6-N3-random`, `8x8-N4`, and `16x16-N8`. A profile is selected by the Host
before a Run and contributes to the environment digest.

Feedback reports success, goal discovery, and collision rate, together with
return, step, truncation, and Policy-failure statistics. `trace.jsonl`
contains a bounded semantic trace for at most four Episodes, retaining the
first 128 and last 32 transitions of long Episodes. It contains no Episode
seed, private Case identity, or Host path.

The packaged baseline builds a relative map from public observations and
replans each step. It moves forward only when no currently visible obstacle
can enter the destination in one upstream motion update. It consumes no
private environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.

