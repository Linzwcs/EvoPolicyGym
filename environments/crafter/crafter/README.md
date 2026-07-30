# EvoPolicyGym Crafter Benchmark

This independently installable distribution adapts
[danijar/crafter](https://github.com/danijar/crafter) `1.8.3` to the public
EvoPolicyGym authoring SDK.

`CrafterBenchmark` runs the canonical reward profile:

- one fresh seeded `64 x 64` procedural world per Episode;
- `64 x 64 x 3` uint8 RGB Policy observations;
- the original 17 discrete Actions;
- up to 10,000 steps per Episode;
- all 22 original achievements;
- the official shifted-geometric achievement success score.

The default Benchmark ID is:

```text
crafter/CrafterReward-v1/achievement-score-v1
```

`CrafterConfig(max_episode_steps=...)` can select a shorter bounded profile for
development. The selected horizon is published in `environment_parameters`,
so EvoPolicyGym gives it a distinct Environment digest. Scores from shortened
profiles are not directly comparable with the canonical 10,000-step profile.

## Policy contract

The Policy receives only the canonical RGB frame as a `TensorValue`. The
upstream global semantic map, player coordinates, raw inventory dictionary,
achievement counters, and Environment seed remain trusted-side information.

Actions must be exact integers from `0` through `16`; invalid Actions are
rejected without advancing Crafter. The packaged `baseline_program()` is a
deterministic wander-and-interact starting point, not a competitive reference
agent.

## Scoring and Feedback

For each achievement, success is the percentage of evaluated Episodes in
which it was unlocked. The scalar score reproduces Crafter's official formula:

```text
exp(mean(log(1 + success_percent))) - 1
```

A Policy failure contributes zero achievement credit and zero return for that
Episode. Feedback reports all 22 success rates, aggregate return and length,
termination and failure counts, and bounded public Artifacts:

- `trace.jsonl`: at most four Episodes and 2,048 transitions per Episode,
  retaining the beginning and end of longer trajectories;
- `episode-N-frames.png`: one `1024 x 1024` contact sheet for each of the first
  four Episodes, using up to 16 uniformly sampled or achievement-bearing public
  RGB observations enlarged with nearest-neighbor resampling;
- `episode-0-replay.mp4`: an H.264 replay of up to 300 public observations from
  the first retained Episode at `512 x 512` and 10 FPS;
- `artifact-manifest.json`: the step, Action-independent observation index,
  achievement events, dimensions, and replay-frame mapping for every retained
  visual artifact.

All Episodes participate in scoring even when detailed traces are omitted.
Neither Artifact contains Environment seeds, Policy seeds, Host paths,
scenarios, process evidence, or privileged Crafter state.

The original Policy observation remains `64 x 64`. Contact sheets and replay
frames use integer nearest-neighbor enlargement for readability; they do not
claim additional simulator detail.

## Upstream and license

Crafter is Copyright 2021 Danijar Hafner and distributed under the MIT
License. This package depends on the published `crafter==1.8.3` distribution;
it does not vendor or modify Crafter source code or assets.

Crafter 1.8.3 stores per-chunk objects in address-hashed sets, whose iteration
order can vary between Python processes and change later creature balancing.
After each reset, the adapter replaces only those empty-or-populated chunk
containers with insertion-ordered, set-compatible containers. Existing
objects retain upstream creation order, and all game rules and random draws
remain upstream-owned. A compatibility guard fails closed if the pinned
internal representation changes.

## Development

From this directory:

```console
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

The direct Evaluation test uses `ProcessExecution.unsafe()`. It is not a
sandbox; the trusted packaged baseline runs with the current operating-system
user's authority.

Agent choice, Run coordination, execution settings, workspace management, and
CLI presentation remain outside this Benchmark distribution.
