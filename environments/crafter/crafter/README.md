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

`CrafterConfig(max_episode_steps=..., include_mp4_feedback=...)` can select a
shorter bounded profile and independently enable derived MP4 training feedback.
MP4 is disabled by default. These settings are published in
`environment_parameters`,
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
reported as a diagnostic but is not scored. First achievement
unlocks use a public absolute dependency-stage schedule from `1` through
`1024`; the complete 22-achievement schedule is worth at most `1829` per
Episode. Repeated confirmed drinking, eating, gathering, combat, planting, and
construction events add at most `25` further points using public event weights,
caps, and logarithmic diminishing returns. Attempts do not score unless Crafter
increments the corresponding event counter.

The shaped delta is the Environment `Step.reward`, while the pinned upstream
reward remains available as `Step.metrics["upstream_reward"]`. A completed
Episode therefore has the exactly reconstructable return:

```text
survival + vital + first-unlock progress + repeated productivity
```

A Policy failure instead returns `-max_episode_steps` and discards partial
credit. Natural death merely ends future earning; it does not erase earlier
legitimate progress. `Feedback.score` is the arithmetic mean Episode return.
Feedback publishes the component reconstruction, Episode return variance, standard
deviation, standard error, normal-approximation 95% confidence interval,
`survival@300/600/900`, weakest-vital exposure, terminal vital profile,
achievement/event detail, canonical Crafter comparison, and unscored Action
cycle diagnostics. The complete formula and tables are recorded in
[`docs/long-horizon-feedback-v3-design.md`](docs/long-horizon-feedback-v3-design.md).

## Complete training evidence

For training submissions of at most 64 Episodes, all three profiles encode
every public transition and publish every Policy observation as lossless NPZ.
There is no score-based Episode selection, first-Episode preference, temporal
frame sampling, contact sheet, or hidden human-observer channel. The NPZ stream
is always the primary Agent-facing visual evidence and is byte-exact with the
`TensorValue.data` received by the Policy; the trajectory preserves the exact
Action/reward chronology.

When `CrafterConfig(include_mp4_feedback=True)` is selected, the same feedback
also contains one `replays/episode-N/replay.mp4` per Episode. Each replay covers
all `steps + 1` observations in order, uses 10 FPS playback, nearest-neighbor
scaling to 256 x 256, H.264/YUV420 encoding with a 96 kbit/s target and
112 kbit/s maximum video rate, and no audio or overlays. MP4 is a lossy derived
viewing aid; it never replaces, samples, or changes the lossless NPZ evidence.
The local launcher exposes this switch as
`--include-mp4-feedback` and leaves it off when the flag is absent.

The published layout is:

```text
artifacts/
├── artifact-manifest.json
├── trajectories/
│   ├── episode-000000/trajectory-000000.jsonl.gz
│   └── episode-000001/trajectory-000000.jsonl.gz
├── observations/
│   ├── episode-000000/observations-000000.npz
│   └── episode-000001/observations-000000.npz
└── replays/                                      # only when enabled
    ├── episode-000000/replay.mp4
    └── episode-000001/replay.mp4
```

Each gzip JSONL trajectory contains an Episode header followed by every
transition in order. A transition records the Agent-visible Episode ordinal,
step and observation indices, Action and Action name, reward, first-unlock and
successful-event information, and termination flags. v3 additionally records
the four shaped-reward components and the separate upstream reward. It never records an
Environment seed, Policy seed, pool identity, Host path, process evidence, or
privileged Crafter state.

`artifact-manifest.json` is compact permanent metadata that lists every NPZ
chunk, its Episode-local observation-index range and compressed size, the complete
Episode/transition/frame counts, and the alignment contract. Complete
trajectories are also permanent; their small compressed size preserves the
action/reward history even after old observation chunks expire.

Each Episode's NPZ files contain all `steps + 1` observations in order. Every
chunk has `observations` (`uint8 [N, 64, 64, 3]`) and
`observation_indices` (`uint32 [N]`). Transition `t` therefore remains aligned
as `observation[t] -> action[t] -> observation[t + 1]`. Chunks contain at most
1,024 consecutive observations, do not cross Episode boundaries, and use
lossless ZIP compression without resizing, cropping, overlays, seeds, or private
state. Read them without object deserialization:

```python
import numpy as np

with np.load(path, allow_pickle=False) as data:
    frames = data["observations"]
    indices = data["observation_indices"]
```

The Crafter uv environment includes NumPy and Pillow. NumPy is declared in this
independent distribution's `pyproject.toml` under `[project].dependencies` and
is locked by the adjacent `uv.lock`; it is not a Kernel dependency. Packages
required only by future Agent-side analysis tools belong in this distribution's
`[project.optional-dependencies].agent-tools` extra, following the same
package boundary. Development-only linters and type checkers remain in `dev`.

The launcher tells the Agent to run analysis scripts with `python` directly.
That command resolves to the Python interpreter from the uv environment used to
start the launcher and can import NumPy and Pillow. A bare `uv run python` from
the Run workspace is not equivalent: uv walks upward, discovers the repository
root Kernel project, and selects its intentionally minimal environment, where
NumPy is absent. The Agent may select frames, build contact sheets, or create
other derived visual analyses under `workspace/analysis/`; the Benchmark does
not choose a visualization or impose one on the optimization process.

The Agent can address only opaque Run-local training Episode indices, never
Environment or Policy seeds. The Host plans one fixed pool before the Run, and
every selected index consumes one Episode budget unit. Exact allocation remains
part of the Agent's optimization behavior rather than a Crafter gameplay rule.

Recommended Crafter launch instructions must describe legal selectors and
budget accounting neutrally. They should not explicitly suggest that the Agent
repeatedly evaluate the Episode associated with any particular index. In other
words: 建议启动命令不要明示 Agent 重复使用某一个 index 对应的 Episode。
This documentation rule does not remove any operation allowed by EvoPolicyGym;
it avoids adding a Crafter-specific optimization bias to the startup task.

Validation and Assessment remain Host-only aggregate phases and publish no
detailed evidence to the Agent workspace. Their Host reports retain the same
Benchmark-defined aggregate Feedback content, including the survival profile,
but never construct or copy trajectory, NPZ, or MP4 Artifacts, regardless of
their Episode count. The private Episode plan carries only a Boolean Artifact
mode to `feedback()`; neither the split nor that marker crosses the Policy or
Agent boundary. Train evaluations larger than the documented 64-Episode
detailed-feedback limit likewise return complete aggregate metrics without
constructing per-Episode Artifact files. These rules avoid unnecessary Host
work and the Kernel's 1,024-Artifact bound without selecting or sampling
particular Episodes, and do not change scoring.

The 64-Episode submission ceiling is Crafter-specific, not an EvoPolicyGym
protocol rule. At the 10,000-step horizon, each Episode needs one trajectory
Artifact, at most ten 1,024-observation NPZ chunks, and at most one optional
MP4. A full 64-Episode submission therefore has at most
`64 * (1 + 10 + 1) + 1 = 769` Artifacts including the manifest, below the
Kernel's 1,024-Artifact Feedback limit. The raw RGB
payload in the theoretical all-Episodes-survive-to-horizon case is about
7.3 GiB before compression, however, so the Artifact-count proof is not a
1-GiB storage guarantee. The newest submission remains protected when it alone
exceeds the configured bulk capacity; operators must calibrate capacity from
representative optimized Policies.

In a 2026-08-07
Terra pilot, a 128-Episode submission took about 115 seconds. The Codex command
execution stopped remaining synchronously attached while the Session command
continued in the background; the Agent then queued additional unchanged
submissions before detecting the live processes. The 64-Episode ceiling is a
conservative feedback-cadence mitigation for that observed integration
failure. It cannot guarantee a short wall time when stronger Policies survive
longer, so the caller must still provide a command environment that preserves
long-running synchronous Session calls. The Agent chooses each non-empty batch
size up to this limit.

## Temporal evidence-retention protocol

Lossless observation NPZ files and optional MP4 replays are classified as
`bulk`; complete trajectories, scores,
`feedback.json`, Episode summaries, hashes,
`artifact-manifest.json`, and availability metadata are permanent. The Run
applies one configurable capacity to the actual bytes occupied by bulk files
across both copies:

```text
RUN/submissions/...                         # formal Host record
RUN/workspace/feedback/submissions/...      # Agent-visible mirror
```

After a new submission is published to both locations, capacity enforcement
walks older submission IDs in chronological order. It removes only old
observation NPZ and MP4 bulk files from both views;
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

The Agent owns `workspace/analysis/` for selected frames, derived
summaries, diagnostic scripts, and other working material. This directory is not part of
the submitted Program and is never pruned by bulk retention. The Benchmark,
not the Agent, chooses the lossless chunk format; the Agent chooses what to
inspect and what derived evidence to preserve in `analysis/` before a later
submission makes older bulk data eligible for eviction. Derived files are
Agent-owned analysis; they do not pin or modify the original read-only
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
horizon produced 2,694 transitions and 2,710 observations. Its Episodes ended
after 47–242 steps, so it is not a long-survival storage calibration. The
1-GiB default deliberately leaves substantial headroom for current Policies;
it must be revisited using `bulk_compressed_bytes` from representative
optimized submissions.

## Recommended model-comparison protocol

The current cost-balanced formal Crafter comparison uses:

```text
train Episodes:       1,024
train pool:           1,024 fixed Run-local indices
Validation Episodes: 256 per candidate
test Episodes:        512
Run seed:             one explicit fixed value shared by every model
submission size:      Agent-selected; hard maximum 64, ordinarily about 32 Episodes
candidate evidence:   normally at least 64 Episodes for an exact Program revision
Episode allocation:   Agent-selected; each train index has at most 2 uses by default
finish policy:        early finish allowed; 1,024 Episodes is an upper bound
historical Policies:  Host-retained, never Agent-visible
```

The 1,024-Episode train budget gives a Coding Agent room for early visual
diagnosis and later Policy refinement; it is a fixed comparison budget, not a
claim that learning saturates exactly at 1,024. The Agent remains responsible
for choosing each submission size because evidence allocation and feedback
cadence are part of the end-to-end optimization behavior. The formal launcher
defaults to `allow_early`: an Agent may finish when it no longer expects another
evaluation to justify a Policy change, and every report must include actual
Episode consumption. This keeps unused evidence budget from being converted
into unchanged submissions. Controlled stress tests may opt in to
`require_budget_exhaustion`, but that is not the default comparison protocol.

The 64-Episode submission limit is a Host safety and Artifact-size bound, not
the preferred size of every evaluation. Local retained-run measurements found
per-Episode score standard deviations around 55--62 under the raw-return
profile. At that noise level, an 8-Episode mean has an approximate 95% sampling
half-width near 42 score points, compared with about 21 at 32 Episodes and 15
at 64 Episodes. Therefore the formal launcher recommends about 32 Episodes for
an ordinary comparison, reserves smaller batches for smoke tests or targeted
diagnosis, and asks the Agent to accumulate at least 64 total Episode results
for an exact submitted Program revision before rejecting a promising direction
or concluding that further improvement is unjustified. Those 64 results may
span submissions. This is recorded statistical guidance rather than a Kernel
quota: the Agent still controls evidence allocation and may finish early.

Comparisons should combine deliberate matched-index evidence with fresh unseen
indices. A matched comparison is valid only for the exact same submitted
Program revision; manually recreating or approximately reverting source creates
a different Policy even when the intended behavior is similar.

The evaluation sizes follow a local five-panel audit of three retained
Policies under this raw-return metric. Moving from test32 to test256 reduced
empirical finite-pool variance by 76.6%--86.4%. Validation256 and test512
retain that sampling margin, and Feedback publishes uncertainty statistics for
every evaluated batch. The sizes reduce procedural-world sampling noise; they
do not remove variation between independent Coding-Agent optimization runs.

All compared models must receive the same explicit Run seed, Environment
configuration, horizon, scoring profile, initial Program, limits, and Agent
instructions. Their fixed Run-local train pools, Validation Episodes, and test
Episodes are then exactly paired. Agents may nevertheless select different
train indices, so equal Episode consumption does not imply equal unique rollout
coverage. Reports must include both consumed Episode units and the number of
distinct train indices used; this is intentional end-to-end optimization
behavior, not identical training exposure.

Historical Policy source is excluded from Agent Feedback. Earlier controlled
Runs showed a directional advantage for the no-history arm, while one Run per
condition was insufficient for a causal claim. Exclusion is nevertheless the
cleaner default: the current Program remains available, Host provenance is
unchanged, and the workspace avoids an unproven source of restoration bias and
attention anchoring. Any future history ablation should be a separate repeated
experiment rather than a switch in the formal comparison.

### Reproducible uv and Python environment

From the repository root, one protocol-conforming survival-development Run can
be launched with:

```console
UV=/data/home/lilianhsong/.local/bin/uv
$UV sync \
  --project environments/crafter/crafter \
  --extra dev \
  --locked
```

The reproducible tool-environment contract has three distinct stages:

1. The Host runs the locked sync above against Crafter's own `pyproject.toml`
   and `uv.lock`; an unlocked sync or the repository-root Kernel project is not
   an equivalent environment.
2. The Host starts the launcher with Crafter's `.venv/bin/python`, as below, or
   with `uv run --project environments/crafter/crafter --locked`. It must not
   start the launcher through an unrelated Python installation.
3. Inside the Run workspace, the Agent invokes `python` directly. The Kernel
   places the launcher's interpreter directory first on the Agent `PATH`, so
   this is the same uv-managed environment. Bare `uv run` from the workspace is
   prohibited because it discovers the repository-root Kernel project.

This produces one Crafter Agent work-and-analysis environment: the CLI, NumPy,
and Pillow are available to policy-development scripts. Codex is started with
its local-image viewing tool enabled, so it can extract selected PNG frames or
contact sheets from the lossless NPZ observations and inspect those images.
The separate process lifecycle preserves logical Benchmark ownership and
publication boundaries for compliant code; it does not enforce secrecy or
operating-system isolation against a hostile Agent or Policy.

```console
environments/crafter/crafter/.venv/bin/python \
  environments/crafter/crafter/scripts/run_crafter_codex.py \
  --model gpt-5.6-sol \
  --record-to runs/crafter-survival-development1024-sol-<run-id> \
  --profile survival-development \
  --max-episode-steps 10000 \
  --seed 20260804 \
  --max-submissions 1024 \
  --episode-budget 1024 \
  --max-episodes-per-submission 64 \
  --recommended-episodes-per-submission 32 \
  --minimum-candidate-evidence 64 \
  --max-train-index-uses 2 \
  --finish-budget-policy allow_early \
  --bulk-feedback-retention-bytes 1073741824 \
  --validation-episodes-per-candidate 256 \
  --validation-max-candidates 3 \
  --assessment-episodes 512 \
  --episode-timeout-seconds 600 \
  --agent-timeout-seconds 43200 \
  --progress plain \
  --allow-unsafe-process
```

Add `--include-mp4-feedback` to that launcher command when the experiment should
publish MP4 alongside NPZ. Omit it (the default), or pass
`--no-include-mp4-feedback`, for NPZ-only feedback.

The packaged Benchmark skill stays disabled unless `--benchmark-skill` is
passed. `--codex-executable` can select a caller-owned Codex executable or
wrapper, but command selection alone does not isolate the whole Run or the
Policy processes that the Host creates. By default,
`--max-train-index-uses 2` adds a launcher-level Codex instruction that permits
each Run-local train Episode index to be selected at most twice: its first
evaluation plus at most one deliberate matched-Policy retry. While unseen
indices remain, each new submission must include unseen indices, and Codex is
asked to maintain an index-usage record under `analysis/`. Set another positive
integer externally, for example `--max-train-index-uses 3`, to change the
limit, or pass `--max-train-index-uses none` to omit the training-index
instruction entirely.
This is an auditable Agent instruction retained in `agent/invocation.json`, not
a Kernel-enforced rejection rule; reports must still calculate actual unique
index coverage from Run records.

The companion `--recommended-episodes-per-submission 32` and
`--minimum-candidate-evidence 64` options publish the sampling guidance above
in the same recorded invocation. Both accept another positive integer or
`none`; the recommendation cannot exceed the hard per-submission maximum, and
the minimum evidence cannot exceed the whole Run's Episode budget. Disabling
either option removes only that guidance, not the Benchmark limit or scoring
logic.

Keep `--record-to` short enough that its absolute
`control/session.sock` path is below Linux's 108-byte Unix-domain socket limit.
The launcher rejects longer paths before creating the Run; concise identifiers
under `runs/` are recommended.

The same recorded launcher instruction identifies `python` as the supported
Agent analysis interpreter. In particular, it warns against bare `uv run` from
the Run workspace because that command discovers the Kernel project rather
than this independent Crafter distribution.

Neither launcher instruction changes `finish_budget_policy`. The index-use
instruction does not prescribe a selector, and the batch-evidence instruction
states recommendations rather than enforcing submission sizes. The default
launcher treats `episode_budget` as an upper bound and permits an earlier
successful finish.

### Execution isolation

EvoPolicyGym 0.3 exposes only `ProcessExecution.unsafe()` for Evaluation and
Runs. `--allow-unsafe-process` acknowledges that choice; it does not enable a
sandbox. The profile does not provide namespace, seccomp, cgroup, container,
microVM, CPU, memory, PID, disk, descriptor, or network confinement, and both
Agent and Policy processes retain the authority of the current operating-system
user. The separately installable Firecracker package is alpha groundwork, not
a qualified isolation profile.

Consequently, local `ProcessExecution.unsafe()` is suitable only for trusted
Agent, Program, and Benchmark code. If Host protection or a formal source-access
boundary is required, the caller must start the entire Crafter launcher inside
a caller-owned container, virtual machine, or remote workload boundary. That
boundary should use an unprivileged identity, expose only required read-only
inputs and the intended Run output, omit Host credentials and unrelated source,
and independently constrain network, CPU, memory, processes, and disk. Wrapping
only the Codex executable is insufficient because Policy evaluation remains a
Host-created local subprocess. A future formal execution profile must be
selected explicitly and must never silently fall back to
`ProcessExecution.unsafe()`.

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
