# MiniGrid PutNear Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid PutNear](https://minigrid.farama.org/environments/minigrid/PutNearEnv/)
object-rearrangement task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and a relational mission. It must pick up exactly the named
colored object and drop it in an empty cell next to a second named object.
Picking up a wrong object terminates immediately. Every drop attempted while
carrying also terminates: an actual adjacent drop succeeds, an actual
non-adjacent drop is misplaced, and a drop into an occupied cell is a distinct
blocked-drop failure that leaves the object carried. The primary score is
Episode success rate.

Although upstream documentation labels `toggle` unused, toggling an empty box
destroys it without termination. Destroying either named mission object makes
the task impossible; Feedback diagnoses this explicitly.

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

Feedback reports object labels and discovery steps, correct and wrong pickups,
whether the front cell is empty and adjacent to the target, valid-drop
availability, misplaced and blocked drops, destroyed mission boxes, failed
interactions, observation novelty, ineffective actions, action mix, exact
terminal reason, return, truncation, and Policy failures. `trace.jsonl`
contains a bounded semantic trace for at most four Episodes and no Episode
seed, private identity, or Host path.

The packaged baseline parses the public mission, builds a relative map from
public observations, shortest-path plans to the movable object, and selects a
reachable empty drop cell adjacent to the target. It consumes no private
environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.
