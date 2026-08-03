# MiniGrid GoToObject Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid GoToObject](https://minigrid.farama.org/environments/minigrid/GoToObjectEnv/)
instruction-following task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and a natural-language mission identifying a colored key,
ball, or box. It must explore the room, stand next to the requested object, and
issue `done`; an incorrect completion ends the Episode with zero reward.
Executing `toggle` also terminates immediately with zero reward; it is not a
completion alias.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/go_to_object --extra dev
uv run --project environments/minigrid/minigrid/go_to_object \
  python -m unittest discover \
  -s environments/minigrid/minigrid/go_to_object/tests
uv build environments/minigrid/minigrid/go_to_object
```

## Public API

```python
from minigrid_go_to_object import GoToObjectBenchmark, GoToObjectConfig, baseline_program

benchmark = GoToObjectBenchmark(
    GoToObjectConfig(profile="8x8-N2"),
)
program = baseline_program()
```

Available profiles are `6x6-N2` and `8x8-N2`. A profile is selected
by the Benchmark Host before a Run, is included in the environment digest, and
cannot be selected or changed by the Policy.

The spec defines image axes and channels, view orientation, compass and
symbolic encodings, exact reward, and both `toggle` and `done` termination
rules. Feedback reports target discovery and first-seen step, visible and
historically discovered candidate counts, whether the target is visibly
adjacent, the visible adjacent object before completion, remaining horizon,
observation novelty, ineffective Actions, and per-Action usage. Incorrect
completion is separated into premature `toggle` and `done` from a wrong
location. `trace.jsonl` contains a bounded semantic trace for at most four
Episodes, retaining the first 128 and last 32 transitions of long Episodes. It
contains no Episode seed, private Case identity, or Host path.

The packaged baseline builds a relative map from public egocentric
observations, parses the public mission, and shortest-path plans to the matching
object. It consumes no private environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.
