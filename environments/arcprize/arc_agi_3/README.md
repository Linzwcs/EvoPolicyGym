# ARC-AGI-3 Benchmark

This independent distribution exposes complete ARC-AGI-3 interactive games
through EvoPolicyGym's public authoring SPI. One ARC game instance is one
Episode. A Policy keeps its state across every action and level in that game;
`GAME_OVER` can be followed by Action `0`, which resets the current level on
the same official wrapper.

The default `public-25` profile pins the 25 full game version IDs returned by
the official discovery API on 2026-08-01. Pinning full IDs makes the Benchmark
identity stable if ARC Prize later publishes a new version under an existing
four-character slug. A custom fixed collection can include newly discovered or
private games:

```python
from arc_agi_3_benchmarks import ArcAgi3Benchmark, ArcAgi3Config

public = ArcAgi3Benchmark()
custom = ArcAgi3Benchmark(
    ArcAgi3Config(
        profile="custom",
        custom_game_ids=("ls20-9607627b", "ft09-0d8bbf25"),
    ),
    environments_dir="runs/arc-agi-3/environments",
)
```

The Benchmark has one fixed execution model: the official toolkit discovers or
downloads games and runs them locally. API keys and Host paths are
constructor/runtime values; they are never included in `PolicyContext`, Episode
observations, or Feedback. The default runtime asset and recording directories
are under `runs/`.

## Episode and action contract

Each observation contains all animation frames from the last official response
as an `int8` tensor shaped `(frames, 64, 64)`, the public game state, completed
and winning level counts, and the currently available non-reset actions.

Actions are strict PolicyValue objects:

```python
{"action": 1}                    # simple action
{"action": 6, "x": 32, "y": 32}  # coordinate action
{"action": 0}                    # reset the current level
```

Actions `1` through `7` must be present in the current observation's
`available_actions`. Action `6` requires exact integer coordinates from 0
through 63. No invalid action is repaired or replaced.

The primary metric is the official score returned when the shared scorecard is
closed. Per-step reward is the non-negative increase in levels completed; it is
only an optimization signal and is not used as the primary score.

## Detailed training feedback

Feedback contains bounded per-Episode summaries with return, step count, Policy
failure, action counts, final state, completed levels, available actions, frame
count, and a final-frame digest. It also publishes `trace.jsonl` for up to eight
Episodes and 128 steps per traced Episode. Each observation is stored exactly
once with an `observation_index`; transitions contain the exact Action, reward
and termination flags, and reference their decision/result observation indices.
Observation entries include the same game state, level counters, available
actions, and complete `frames` tensor received by the Policy. Frame tensors are
losslessly stored as keyed `int8` arrays in an independent
`episode-NNN/observations.npz` Artifact; the trace retains their dtype, shape,
Artifact name, key, and SHA-256 digest. NumPy can load an Artifact with
`numpy.load(..., allow_pickle=False)`.

Complete frame publication is bounded by per-Episode and total raw-frame byte
caps. When a cap is reached, the trace omits whole trailing steps or Episodes;
it never samples or truncates the animation frames of an observation it
publishes. Each Episode owns its observation Artifact, even when another
Episode contains identical data.

Every Episode with a visual observation publishes `episode-NNN/playback.gif`
using the official ARC-AGI-3 16-color palette. The accompanying `videos`
manifest maps each selected initial/post-Action observation and game state to
its GIF frame range and to the next decision step. A video contains at most 512
frames: every selected observation retains its final decision frame, while
extra animation frames are
uniformly sampled when necessary. Playback uses 100 ms per encoded frame and
nearest-neighbor scaling. A status strip identifies the observation, source
animation frame, decision step, game state, and completed/required levels; the
GIF is diagnostic visualization, not an additional scoring input.

The trace, observation arrays, manifest, and videos deliberately exclude game
ID, scenario, Environment and Policy seeds, scorecard/GUID values, Host paths,
and execution evidence. Additional trace Episodes or steps are reported as
omitted; all Episode results remain in the Kernel-owned `feedback.json`
`episodes` array, and only the diagnostic GIF may sample animation frames.

## Episode seeds

Every Episode receives a deterministic Host-owned seed derived from its split,
Evaluation seed, and Episode index. The seed is passed to `Arcade.make()` and
never crosses the Policy boundary. The toolkit forwards it only when the game
constructor declares a `seed` parameter; games without seeded generation can
therefore have identical initial states despite distinct Episode seeds. A
one-game custom collection can still run multiple fresh instances:

```python
single_game = ArcAgi3Benchmark(
    ArcAgi3Config(
        profile="custom",
        custom_game_ids=("ls20-9607627b",),
    )
)
```

Calling `episodes(..., count=N)` on this Benchmark creates `N` fresh wrappers
for the same game and supplies `N` different seeds. Whether those seeds alter
the game is defined by the upstream game implementation. Runs share one local
official scorecard for the Evaluation. The upstream scorecard selects the best
run for a repeated game; it is not a mean across seeds.

Remote `online` and official `competition` submission behavior are deliberately
outside this Benchmark contract. A future competition runner should own those
server lifecycle rules separately.

```console
uv sync --project environments/arcprize/arc_agi_3 --extra dev
uv run --project environments/arcprize/arc_agi_3 \
  python -m unittest discover -s environments/arcprize/arc_agi_3/tests
uv build environments/arcprize/arc_agi_3
```

The ARC-AGI toolkit and ARCEngine are MIT-licensed upstream projects published
by the ARC Prize Foundation. Game source downloaded by the toolkit remains an
upstream runtime asset and is not included in this distribution.

Upstream references: [ARC-AGI Toolkit](https://github.com/arcprize/ARC-AGI),
[scoring methodology](https://docs.arcprize.org/methodology), and
[scorecards](https://docs.arcprize.org/scorecards).
