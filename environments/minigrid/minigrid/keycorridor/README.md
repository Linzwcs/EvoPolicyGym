# MiniGrid KeyCorridor Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid KeyCorridor](https://minigrid.farama.org/environments/minigrid/KeyCorridorEnv/)
multi-room task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and a mission naming the target ball color. It must explore
multiple rooms, acquire the key matching the locked door, unlock the target
room, free its carrying slot, and pick up the mission-matching ball. The key
and locked-door color match each other; they are independent of the mission
ball color. The primary score is mean upstream Episode return; success rate
remains separately reported.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/keycorridor --extra dev
uv run --project environments/minigrid/minigrid/keycorridor \
  python -m unittest discover \
  -s environments/minigrid/minigrid/keycorridor/tests
uv build environments/minigrid/minigrid/keycorridor
```

## Public API

```python
from minigrid_keycorridor import (
    KeyCorridorBenchmark,
    KeyCorridorConfig,
    baseline_program,
)

benchmark = KeyCorridorBenchmark(
    KeyCorridorConfig(profile="S4R3"),
)
program = baseline_program()
```

Available profiles are `S3R1`, `S3R2`, `S3R3`, `S4R3`, `S5R3`, and `S6R3`.
`S` is the upstream room-size parameter and `R` is the number of room rows;
every profile has three room columns.

The spec defines image axes and channels, view orientation, carried-object and
compass encodings, the exact reward formula, and termination rules. Feedback
reports the Policy-visible progress ladder: target, key, and locked-door colors;
first discovery steps; exploration-door toggles; key pickup/drop; target-door
opening; target approach; current front and carried objects; remaining horizon;
observation novelty; ineffective Actions; and per-Action usage. It explicitly
diagnoses attempts to pick up the target while the key still occupies the
carrying slot. It also includes return, step, truncation, and Policy-failure
statistics. `trace.jsonl` contains a bounded semantic trace for at most four
Episodes, retaining the first 128 and last 32 transitions of long Episodes. It
contains no Episode seed, private Case identity, or Host path.

The packaged baseline builds a relative multi-room map from egocentric
observations, opens accessible doors during exploration, and uses shortest-path
planning between the matching key, locked door, and ball. It consumes no
private environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.
