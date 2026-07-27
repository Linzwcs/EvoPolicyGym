# MiniGrid PutNear Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid PutNear](https://minigrid.farama.org/environments/minigrid/PutNearEnv/)
object-rearrangement task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and a relational mission. It must pick up exactly the named
colored object and drop it in an empty cell next to a second named object.
Wrong pickups and misplaced drops terminate the Episode with zero reward. The
primary score is Episode success rate.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/put_near --extra dev
uv run --project environments/minigrid/minigrid/put_near \
  python -m unittest discover \
  -s environments/minigrid/minigrid/put_near/tests
uv build environments/minigrid/minigrid/put_near
```

## Public API

```python
from minigrid_put_near import (
    PutNearBenchmark,
    PutNearConfig,
    baseline_program,
)

benchmark = PutNearBenchmark(
    PutNearConfig(profile="8x8-N3"),
)
program = baseline_program()
```

Available profiles are `6x6-N2` and `8x8-N3`. A profile is selected by the
Host before a Run and contributes to the environment digest.

Feedback reports success, discovery of both mission objects, wrong pickups,
and misplaced drops, together with return, step, truncation, and
Policy-failure statistics. `trace.jsonl` contains a bounded semantic trace for
at most four Episodes and no Episode seed, private identity, or Host path.

The packaged baseline parses the public mission, builds a relative map from
public observations, shortest-path plans to the movable object, and selects a
reachable empty drop cell adjacent to the target. It consumes no private
environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.

