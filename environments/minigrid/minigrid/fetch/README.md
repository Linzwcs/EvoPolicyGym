# MiniGrid Fetch Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid Fetch](https://minigrid.farama.org/environments/minigrid/FetchEnv/)
instruction-following task.

The Policy receives the upstream `7 × 7 × 3` egocentric symbolic image,
compass direction, and a natural-language mission identifying a colored key or
ball. It must explore the room and pick up exactly the requested object;
choosing a distractor ends the Episode with zero reward. The primary score is
Episode success rate.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/fetch --extra dev
uv run --project environments/minigrid/minigrid/fetch \
  python -m unittest discover \
  -s environments/minigrid/minigrid/fetch/tests
uv build environments/minigrid/minigrid/fetch
```

## Public API

```python
from minigrid_fetch import FetchBenchmark, FetchConfig, baseline_program

benchmark = FetchBenchmark(
    FetchConfig(profile="8x8-N3"),
)
program = baseline_program()
```

Available profiles are `5x5-N2`, `6x6-N2`, and `8x8-N3`. A profile is selected
by the Benchmark Host before a Run, is included in the environment digest, and
cannot be selected or changed by the Policy.

The spec defines image axes and channels, view orientation, the carried-object
encoding, compass and symbolic encodings, the exact reward formula, and natural
termination behavior. Feedback reports the mission target, discovery and
first-seen milestones, when the target first reached the cell in front of the
agent, visible and historically discovered candidates, the object in front
before an attempted pickup, the picked-up or wrong-object label, failed pickup
attempts, remaining horizon, observation novelty, ineffective Actions, and
per-Action usage, together with return, step, truncation, and Policy-failure
statistics.
`trace.jsonl` contains a bounded semantic trace for at most four Episodes,
retaining the first 128 and last 32 transitions of long Episodes. It contains
no Episode seed, private Case identity, or Host path.

The packaged baseline builds a relative map from public egocentric
observations, parses the public mission, and shortest-path plans to the matching
object. It consumes no private environment state.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.
