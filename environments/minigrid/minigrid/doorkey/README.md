# MiniGrid DoorKey Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid DoorKey](https://minigrid.farama.org/environments/minigrid/DoorKeyEnv/)
partially observable navigation task.

The Policy sees the upstream `7 × 7 × 3` egocentric symbolic image, compass
direction, and mission text. It must explore the room, pick up the yellow key,
unlock the yellow door, and reach the green goal. The primary score is Episode
return from the upstream time-decaying reward; success rate remains separately
reported.

The spec defines image axes `[view_x, view_y, channel]`, channel order
`[object, color, state]`, agent position `(3,6)`, view orientation, compass
codes, every symbolic code, Action meanings, horizon, and exact time-decaying
success reward. Forward decreases `view_y`; right increases `view_x`.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/doorkey --extra dev
uv run --project environments/minigrid/minigrid/doorkey \
  python -m unittest discover \
  -s environments/minigrid/minigrid/doorkey/tests
uv build environments/minigrid/minigrid/doorkey
```

## Public API

```python
from minigrid_doorkey import (
    DoorKeyBenchmark,
    DoorKeyConfig,
    baseline_program,
)

benchmark = DoorKeyBenchmark(DoorKeyConfig(profile="8x8"))
program = baseline_program()
```

Available profiles are `5x5`, `6x6`, `8x8`, and `16x16`.

Feedback exposes the full observable funnel: first seeing the key, picking it
up, first seeing and opening the door, seeing the goal, and succeeding. It also
reports the first step of every milestone, where Episodes stalled, remaining
horizon, observation novelty, ineffective Actions, per-Action usage, aggregate
return, truncations, and Policy failures. `trace.jsonl` contains a bounded
semantic trace for at most four Episodes; long traces retain the first 128 and
last 32 transitions. Milestones are derived only from Policy-visible symbolic
observations and actions. Feedback contains no Episode seed, hidden global
position, private Case identity, or Host path.

The packaged baseline constructs a relative map from its egocentric
observations and uses shortest-path planning between the key, door, and goal.
It consumes no private environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.
