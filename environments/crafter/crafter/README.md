# EvoPolicyGym Crafter Benchmark

This independently installable distribution adapts
[danijar/crafter](https://github.com/danijar/crafter) `1.8.3` to the public
EvoPolicyGym authoring SDK.

The distribution exposes three scoring profiles over the same environment:

- `CrafterBenchmark` preserves the canonical achievement evaluation;
- `CrafterLongHorizonBenchmark` makes survival the gate for sustained
  production and new capabilities (legacy v2);
- `CrafterSurvivalDevelopmentBenchmark` uses an additive per-step survival,
  vital-maintenance, first-unlock, and bounded repeated-productivity return.

All three profiles run:

- one fresh seeded `64 x 64` procedural world per Episode;
- `64 x 64 x 3` uint8 RGB Policy observations;
- the original 17 discrete Actions;
- up to 10,000 steps per Episode;
- all 22 original achievements;
- the official shifted-geometric achievement success score as either the
  primary metric or a public comparison metric.

The Benchmark IDs are:

```text
crafter/CrafterReward-v1/achievement-score-v1
crafter/CrafterReward-v1/long-horizon-development-v2
crafter/CrafterReward-v1/mean-survival-development-return-v3
```

`CrafterConfig(max_episode_steps=...)` can select a shorter bounded profile for
development. The selected horizon is published in `environment_parameters`,
so EvoPolicyGym gives it a distinct Environment digest. Scores from shortened
profiles are not directly comparable with the canonical 10,000-step profile.
The long-horizon profile requires at least 900 steps so that its three scored
day-night bands remain reachable.

## Policy contract

The Policy receives only the canonical RGB frame as a `TensorValue`. The
upstream global semantic map, player coordinates, raw inventory dictionary,
raw achievement counters, and Environment seed remain trusted-side
information. Feedback may publish bounded aggregate successful-event counts
derived from achievement counter increments and aggregate low-vital recovery
counts; it does not expose the counters or inventory to the Policy or in the
trajectory Artifact.

Actions must be exact integers from `0` through `16`; invalid Actions are
rejected without advancing Crafter. The packaged `baseline_program()` exposes
one executable modular scaffold in `policy.py`: visual translation, world
memory, exploration, survival, production, combat/defense, and proposal
coordination. Capability modules deliberately return no proposal until they
are developed, and the coordinator assigns no priority to conflicting
proposals. A policy-seeded, nonreversing short movement macro followed by one
interaction keeps the untouched Program evaluable as a fallback. It does not
use a fixed spatial route, blindly repeat interactions, or attempt utilities
without evidence. It is not a competitive reference agent or a claim about
the best Crafter strategy. The starting Program also contains an ordinary
`PLAYER_GUIDE.md` gameplay reference. It is Program documentation, not a
Codex skill.

## Canonical scoring

For each achievement, success is the percentage of evaluated Episodes in
which it was unlocked. The scalar score reproduces Crafter's official formula:

```text
exp(mean(log(1 + success_percent))) - 1
```

A Policy failure contributes zero achievement credit and zero return for that
Episode. Feedback reports all 22 success rates, aggregate return and length,
termination and failure counts, and unscored aggregate Action diagnostics.
The Action diagnostics cover every evaluated Episode and report the Action
distribution, movement share, immediately reversed movement pairs, the longest
alternating reverse run, and repeated short action cycles with periods from one
through eight. They make stationary spam, two-action oscillation, and small
square-route controller loops visible without changing the official Crafter
score.

## Long-horizon development scoring

`CrafterLongHorizonBenchmark` scores each Episode separately. Policy failure
sets all four components to zero. Otherwise, let `L` be the number of updates
after which the player remains alive:

```text
survival = 100 * (
    min(L, 300)
    + 2 * clamp(L - 300, 0, 300)
    + 4 * clamp(L - 600, 0, 300)
) / 2100
```

The first, second, and third 300-step day-night bands therefore receive
increasing weight. `survival@300`, `survival@600`, and `survival@900` remain
separate reported rates.

Maintenance observes only whether health, food, or drink actually increases
after that vital is at or below the published warning threshold of 5. Each
vital is capped at three credited recoveries per Episode with logarithmic
diminishing returns. Raw attempts such as repeatedly using `do` beside water
do not score unless the underlying vital increases.

Innovation is the percentage of the 22 achievement types first unlocked in
that Episode. Productivity uses repeated, upstream-confirmed resource,
construction, farming, and combat events after their first occurrence.
Drinking and eating are excluded because effective restoration is measured by
maintenance instead; repeated tool and utility crafting is also excluded.
Each included category has a fixed cap and logarithmic diminishing returns.

The Episode score is survival-gated:

```text
episode_score = survival * (
    0.70
    + 0.15 * maintenance / 100
    + 0.10 * productivity / 100
    + 0.05 * innovation / 100
)
```

The scalar Feedback score is the mean Episode score. It can never exceed the
survival component. The canonical Crafter score and all 22 success rates remain
in Feedback for comparison but do not select the long-horizon Program.

## Additive survival-development scoring

`CrafterSurvivalDevelopmentBenchmark` is the current long-horizon optimization
profile. Every transition after which the player remains alive earns `1`
survival point plus:

```text
0.10 * min(health, food, drink) / 9
```

The naturally terminal transition earns neither term. The raw energy meter is
reported as a diagnostic but is not scored. First achievement unlocks use a
public absolute dependency-stage schedule from `1` through `1024`; the complete
22-achievement schedule is worth at most `1829` per Episode. Repeated confirmed
drinking, eating, gathering, combat, planting, and construction events add at
most `25` further points using public event weights, caps, and logarithmic
diminishing returns. Attempts do not score unless Crafter increments the
corresponding event counter.

The shaped delta is the Environment `Step.reward`, while the pinned upstream
reward remains available as `Step.metrics["upstream_reward"]`. A completed
Episode therefore has the exactly reconstructable return:

```text
survival + vital + first-unlock progress + repeated productivity
```

A Policy failure instead returns `-max_episode_steps` and discards partial
credit. Natural death merely ends future earning; it does not erase earlier
legitimate progress. `Feedback.score` is the arithmetic mean Episode return.
Feedback publishes the component reconstruction, Episode return distribution,
`survival@300/600/900`, weakest-vital exposure, terminal vital profile,
achievement/event detail, canonical Crafter comparison, and unscored Action
cycle diagnostics. The complete formula and tables are recorded in
[`docs/long-horizon-feedback-v3-design.md`](docs/long-horizon-feedback-v3-design.md).

## Complete training evidence

For training submissions of at most 16 Episodes, all three profiles encode
every public transition and every RGB observation from every Episode. They also
publish a directly viewable MP4 for every Episode. There is no score-based
Episode selection, first-Episode preference, temporal sampling, contact sheet,
or hidden human-observer channel.

The published layout is:

```text
artifacts/
├── artifact-manifest.json
├── trajectories/
│   ├── episode-000000/trajectory-000000.jsonl.gz
│   └── episode-000001/trajectory-000000.jsonl.gz
├── replays/
│   ├── episode-000000/replay-000000.mp4
│   └── episode-000001/replay-000000.mp4
└── bulk/
    ├── observations-000000.npz
    └── observations-000001.npz
```

Each gzip JSONL trajectory contains an Episode header followed by every
transition in order. A transition records the Agent-visible Episode ordinal,
step and observation indices, Action and Action name, reward, first-unlock and
successful-event information, and termination flags. v3 additionally records
the four shaped-reward components and the separate upstream reward. It never records an
Environment seed, Policy seed, pool identity, Host path, process evidence, or
privileged Crafter state.

Observation chunks contain at most 1,024 frames and can be loaded without
pickle:

```python
import numpy as np

with np.load("observations-000000.npz", allow_pickle=False) as data:
    frames = data["observations"]          # uint8 [N, 64, 64, 3]
    episode_ids = data["episode_indices"] # uint32 [N]
    frame_ids = data["observation_indices"] # uint32 [N]
```

The frames are byte-identical to the RGB `TensorValue.data` received by the
Policy: no resizing, cropping, labels, overlays, or lossy encoding. Indices
preserve every relation
`observation[t] -> action[t] -> observation[t + 1]`.
`artifact-manifest.json` is compact permanent metadata that lists every chunk,
its frame range and compressed size, the complete Episode/transition/frame
counts, and the alignment contract. Complete trajectories are also permanent;
their small compressed size preserves the action/reward history even after old
lossless observation chunks expire.

Each Episode replay contains all `steps + 1` observations in order. Video frame
`i` is observation `i`; therefore transition `t` remains aligned as
`frame[t] -> action[t] -> frame[t + 1]`. The default presentation is H.264 MP4,
10 FPS, 256 x 256, nearest-neighbor expansion from the source image, with no
audio, text, border, coordinate overlay, seed, or private state. Replay FPS and
size are public `CrafterConfig` presentation settings. Episodes longer than
2,048 observations use consecutive MP4 segments listed in the manifest so each
Artifact remains within the Kernel's per-file limit. MP4 is a convenient lossy
browsing layer; NPZ remains the byte-exact observation evidence.

The Agent cannot address seeds or stable cases. `submit --episodes N` consumes
the next `N` Episodes from the hidden train pool and exposes only submission and
Episode ordinals. Submitting an unchanged Program evaluates new Episodes; this
format does not introduce same-seed replay or a private identity side channel.
Validation and Assessment remain Host-only aggregate phases and publish no
detailed evidence to the Agent workspace. Their Host reports retain the same
Benchmark-defined aggregate Feedback content, including the survival profile,
but never construct or copy trajectory, NPZ, or MP4 Artifacts, regardless of
their Episode count. The private Episode plan carries only a Boolean Artifact
mode to `feedback()`; neither the split nor that marker crosses the Policy or
Agent boundary. Train evaluations larger than the documented 16-Episode
detailed-feedback limit likewise return complete aggregate metrics without
constructing per-Episode Artifact files. These rules avoid unnecessary Host
work and the Kernel's 1,024-Artifact bound without selecting or sampling
particular Episodes, and do not change scoring.

## Temporal evidence-retention protocol

Lossless RGB chunks and MP4 replays are classified as `bulk`; complete
trajectories, scores, `feedback.json`, Episode summaries, hashes,
`artifact-manifest.json`, and availability metadata are permanent. The Run
applies one configurable capacity to the actual bytes occupied by bulk files
across both copies:

```text
RUN/submissions/...                         # formal Host record
RUN/workspace/feedback/submissions/...      # Agent-visible mirror
```

After a new submission is published to both locations, capacity enforcement
walks older submission IDs in chronological order. It removes only old
observation NPZ and replay MP4 bulk files from both views;
it does not remove whole submission records or compact Feedback. The newest
successfully published submission is always protected. If that submission
alone exceeds the configured capacity, the data stays complete and
`workspace/feedback/retention.json` reports
`over_limit_to_preserve_latest: true`.

Each submission retains `availability.json`, so a missing old bulk file is an
explicit `evicted` state rather than silent corruption. Original artifact
names, sizes, and SHA-256 hashes remain in `feedback.json` and the availability
document. Retention failures are reported in `retention.json` and do not turn a
successfully evaluated submission into a Policy failure.

The Agent owns `workspace/analysis/` for selected frames, derived summaries,
diagnostic scripts, and other working material. This directory is not part of
the submitted Program and is never pruned by bulk retention. The Benchmark,
not the Agent, chooses the lossless wire compression; the Agent chooses what to
decode and what derived evidence to preserve in `analysis/` before a later
submission makes older bulk data eligible for eviction. To keep a replay, copy
it to a path that retains its public association, for example
`analysis/selected-replays/submission-000008/episode-000000.mp4`. This creates
Agent-owned analysis; it does not pin or modify the original read-only
Feedback Artifact.

Formal `RUN/submissions/SUBMISSION_ID/program/` snapshots are always retained
for Host audit and human review. They are never mirrored into Agent-visible
Feedback. The Agent develops only the current editable `workspace/program/`
and may preserve its own derived notes under `workspace/analysis/`. This keeps
historical source provenance without making earlier Policies an implicit
optimization hint or anchoring point.

The current default capacity is 1 GiB counted across both physical copies. It
is a provisional operating value, not a benchmark constant. Calibrate it from
representative runs using each submission's `bulk_compressed_bytes` and the
central `retention.json`; set it high enough for the expected newest complete
submission plus the desired amount of recent history. Raising or lowering this
storage value does not change scoring, Episode assignment, or Policy behavior.

On 2026-08-02, a local 16-Episode baseline measurement at the 10,000-step
horizon produced 2,694 transitions and 2,710 observations. That measurement
predated public MP4 publication and its Episodes ended after 47–242 steps, so
it is not a current storage calibration. The 1-GiB default deliberately leaves
substantial headroom; it must be revisited using
`bulk_compressed_bytes` from representative optimized submissions.

## Recommended model-comparison protocol

The current cost-balanced formal Crafter comparison uses:

```text
train Episodes:       1,024
Validation Episodes: 256 per candidate
test Episodes:        512
Run seed:             one explicit fixed value shared by every model
submission size:      Agent-selected, at most 16 Episodes
historical Policies:  Host-retained, never Agent-visible
```

The 1,024-Episode train budget gives a Coding Agent room for early visual
diagnosis and later Policy refinement; it is a fixed comparison budget, not a
claim that learning saturates exactly at 1,024. The Agent remains responsible
for choosing each submission size because evidence allocation and feedback
cadence are part of the end-to-end optimization behavior. It is an upper bound:
the Agent may finish earlier, and every comparison must report the actual
Episodes consumed.

The evaluation sizes follow a local five-panel audit of three retained
Policies. Moving from test32 to test256 reduced empirical finite-pool variance
by 76.6%--86.4%, but 256 remained close to the desired cross-pool mean-SD
boundary of ten score points. Validation256 and test512 therefore add margin
for candidate selection and final reporting. This reduces procedural-world
sampling noise; it does not remove stochastic variation between independent
Coding-Agent optimization trajectories.

All compared models must receive the same explicit Run seed, Environment
configuration, horizon, scoring profile, initial Program, limits, and Agent
instructions. Their Validation and test Episodes are then exactly paired.
Train Episode planning is submission-scoped, so Agents that choose different
submission-size sequences may receive different train sequences even with the
same Run seed. This is intentional in the end-to-end protocol and must be
reported rather than mistaken for identical training evidence.

Historical Policy source is excluded from Agent Feedback. Earlier controlled
Runs showed a directional advantage for the no-history arm, while one Run per
condition was insufficient for a causal claim. Exclusion is nevertheless the
cleaner default: the current Program remains available, Host provenance is
unchanged, and the workspace avoids an unproven source of restoration bias and
attention anchoring. Any future history ablation should be a separate repeated
experiment rather than a switch in the formal comparison.

From the repository root, one protocol-conforming survival-development Run can
be launched with:

```console
UV=/data/home/lilianhsong/.local/bin/uv
$UV venv /data/tmp/evopolicygym-agent-tools --python 3.12
$UV pip install \
  --python /data/tmp/evopolicygym-agent-tools/bin/python \
  '.[agent-tools]' \
  'imageio>=2.37,<3' \
  'imageio-ffmpeg>=0.6,<0.7'
```

This is one Agent work-and-analysis environment: the CLI, NumPy, Pillow,
ImageIO, and imageio-ffmpeg are all available to policy-development scripts.
The Crafter package keeps the
same NumPy/Pillow guarantees for formal Policy execution, while Codex is
started with its local-image viewing tool enabled. The separate process
boundary protects Benchmark ownership and hidden Episode identity; it no
longer creates an artificial tooling mismatch. The non-editable install also
keeps repository source outside the isolated Agent environment.

```console
environments/crafter/crafter/.venv/bin/python scripts/run_crafter_codex.py \
  --model gpt-5.6-sol \
  --record-to runs/crafter-survival-development1024-sol-<run-id> \
  --profile survival-development \
  --max-episode-steps 10000 \
  --seed 20260804 \
  --max-submissions 1024 \
  --episode-budget 1024 \
  --max-episodes-per-submission 16 \
  --bulk-feedback-retention-bytes 1073741824 \
  --validation-episodes-per-candidate 256 \
  --validation-max-candidates 3 \
  --assessment-episodes 512 \
  --episode-timeout-seconds 600 \
  --agent-timeout-seconds 43200 \
  --progress plain \
  --allow-unsafe-process
```

The packaged Benchmark skill stays disabled unless `--benchmark-skill` is
passed. `--codex-executable` can select the caller-owned isolated Codex wrapper
used by local experiments. The launcher treats `episode_budget` as an upper
bound and records actual consumption. EvoPolicyGym's
`ProcessExecution.unsafe()` remains
explicitly non-sandboxed. The Run directory split enforces data ownership and
publication semantics, but is not an operating-system security boundary; a
formal source-access prohibition still requires caller-owned whole-process
isolation.

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
