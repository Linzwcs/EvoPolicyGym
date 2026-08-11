# EvoPolicyGym Crafter Benchmark

This distribution adapts pinned `crafter==1.8.3` to EvoPolicyGym. It provides
one 10,000-step open-world survival task with two public reward profiles and two
observation profiles.

## Public profiles

Only these reward profiles are supported:

| Launcher value | Benchmark class | Primary metric |
| --- | --- | --- |
| `lhs` (default) | `CrafterLongHorizonSurvivalBenchmark` | `long_horizon_survival_score` |
| `canonical` | `CrafterBenchmark` | `crafter_score_percent` |

The local launcher defaults to LHS. Canonical scoring remains available as
the native Crafter control. Earlier experimental M-series and S-series reward
profiles are not part of the public distribution.

Both classes use deterministic, split-scoped Episode planning. A Run-local
Episode index resolves to the same hidden Episode specification and Policy seed
for a fixed Run seed.

## Policy contract

The action is one exact integer in `[0, 16]`:

```text
 0 noop                 9 place_furnace
 1 move_left           10 place_plant
 2 move_right          11 make_wood_pickaxe
 3 move_up             12 make_stone_pickaxe
 4 move_down           13 make_iron_pickaxe
 5 do                  14 make_wood_sword
 6 sleep               15 make_stone_sword
 7 place_stone         16 make_iron_sword
 8 place_table
```

The authoritative action names are published by `crafter_benchmarks.ACTIONS`;
Policy code should use that ordered contract rather than this visual layout.

Invalid action types or values are Policy failures and do not advance the
simulator. Natural death is termination. Reaching the configured horizon is
truncation. `CrafterConfig.max_episode_steps` accepts `1..10_000` and defaults
to the official 10,000-step horizon.

`program/PLAYER_GUIDE.md` is packaged with both starting Programs. It describes
player-visible game mechanics but does not prescribe a fixed controller.

## Observation profiles

Select the observation independently of reward with
`CrafterConfig.observation_profile` or launcher
`--observation-profile`.

### RGB (default)

The Policy receives a `TensorValue` containing the canonical rendered
observation:

```text
dtype: uint8
shape: [64, 64, 3]
layout: HWC RGB
```

No structured health, inventory, position, semantic map, or seed is exposed to
the Policy.

### Local symbolic v1

`local-symbolic-v1` exposes a player-centered symbolic description of the same
visible local region:

```python
{
    "terrain": TensorValue(dtype="uint8", shape=(7, 9), ...),
    "entities": TensorValue(dtype="uint8", shape=(7, 9), ...),
    "inventory": {...},
    "facing": "down",
    "sleeping": False,
    "daylight": 0.82,
}
```

It uses the same simulator, worlds, dynamics, actions, horizon, Episode pools,
and selected reward profile as RGB. It does not expose a global semantic map,
absolute player position, environment seed, RNG state, achievement counters,
hidden life counters, or entity health/cooldowns. Symbolic Benchmark IDs and
Environment digests are distinct from RGB IDs and digests.

The full symbolic contract and stable IDs are documented in
[`docs/local-symbolic-observation-v1-design.md`](docs/local-symbolic-observation-v1-design.md).
`include_mp4_feedback=True` is intentionally rejected for symbolic Runs.

## Default LHS scoring

LHS is the distribution's default optimization objective. It prioritizes
healthy survival and weak-Episode robustness while retaining bounded progress
and maintenance incentives.

After action `t`, define:

```text
alive_t = 1 unless this is the natural terminal transition, else 0
q_t = min(health_t, food_t, drink_t) / 9

alive_survival_t = 0.01 * alive_t
vital_survival_t = 0.03 * alive_t * q_t
```

The public transition reward is:

```text
Step.reward_t =
    alive_survival_t
    + vital_survival_t
    + first_unlock_delta_t
    + maintenance_repeat_delta_t
    + productivity_repeat_delta_t
```

First-unlock credit is `0.10 * log2(1 + raw_weight)`. Productive repeats
receive at most 20% of the corresponding first-unlock credit in a rolling
300-step window, spread over event-specific quotas. Maintenance credit is
limited to actual possible food/drink restoration and rolling restoration-unit
caps. A first successful event is never also counted as a repeat.

At full visible vitals, one alive transition is worth `0.04` and a healthy
300-step day is worth 12 survival points. Energy and native Crafter reward are
diagnostics, not LHS score components.

For Episode `i`:

```text
survival_return_i = sum(alive_survival + vital_survival)
secondary_return_i = sum(first_unlock + maintenance_repeat + productivity_repeat)
episode_return_i = survival_return_i + secondary_return_i
```

A Policy execution failure has formal return zero. Its partial trace and
partial component totals remain visible and are marked as discarded.

Across `N` Episodes, let `k = max(1, ceil(0.25 * N))` and select the lower tail
using survival return alone:

```text
LHS_score =
    0.75 * mean(survival_return)
    + 0.25 * mean(bottom-k survival returns)
    + mean(secondary_return)
```

There is no upper-tail bonus. The Feedback separately reports canonical
Crafter score, achievement success, effective-survival distribution, survival
rates at 150/200/250/300/400 steps, vital quality by Episode age, terminal
food/drink state, native return, health changes, and action diagnostics.

The exact constants and reconstruction contract are documented in
[`docs/long-horizon-survival-score-design.md`](docs/long-horizon-survival-score-design.md).

## Canonical scoring

Canonical Feedback follows Crafter's shifted geometric mean over the 22
achievement success percentages:

```text
C = exp(mean(log(1 + success_percent_i))) - 1
```

Each achievement contributes whether it was completed at least once in an
Episode. Repeated completion does not increase `C`. Policy-failed Episodes
receive no achievement credit. Native transition reward and mean return remain
public diagnostics.

## Training Feedback and Artifacts

Detailed Artifacts are generated only for training Feedback containing at most
64 Episodes. The limit bounds one submission under the Kernel Artifact-count
ceiling while allowing a statistically useful batch. If a caller evaluates
more than 64 Episodes in one submission, scoring remains complete but Feedback
is aggregate-only.

For each detailed training Episode, the Environment publishes:

```text
trajectories/episode-XXXXXX/trajectory-000000.jsonl.gz
observations/episode-XXXXXX/observations-XXXXXX.npz
```

The trajectory is complete and aligned as:

```text
observation[t] -> action[t] -> observation[t + 1]
```

RGB NPZ chunks contain byte-exact `uint8 [frames, 64, 64, 3]` observations and
`uint32` observation indices. Symbolic NPZ chunks losslessly contain terrain,
entities, inventory, facing, sleeping, daylight, and observation indices. Each
chunk contains at most 1,024 observations.

With RGB and `--include-mp4-feedback`, the same submission additionally
contains one derived H.264 MP4 per Episode. MP4 is a viewing aid; lossless NPZ
remains authoritative. Frames are not temporally sampled. MP4 is unavailable
for local symbolic observations.

Validation and held-out test never publish trajectories, NPZ, or MP4. They
return aggregate Feedback only.

Trajectories have permanent retention. NPZ and optional MP4 use bulk retention.
The Run's `bulk_feedback_retention_bytes` capacity applies to both Host records
and Agent-workspace mirrors: older bulk files are removed first while the
newest submission remains protected. Structured Feedback and permanent
trajectories are not bulk-evicted.

## Recommended Run configuration

The documented model-comparison configuration is:

```text
train Episode budget: 1024
validation:           256 Episodes per candidate
held-out test:        512 Episodes
Run seed:             one fixed value shared across compared models
submission maximum:  64 Episodes
```

The larger validation and test pools reduce the measured variance of a fixed
Policy. The 64-Episode submission maximum is an Artifact-count/storage bound,
not an EvoPolicyGym-wide protocol rule. The Agent may choose smaller
submissions.

The launcher adds a caller-owned diversity condition: a training index may be
used at most once during that Run. Launch instructions should not encourage
repeated reuse of one index. This is an experimental condition, not a Kernel
change.

Early finish is the default (`--finish-budget-policy allow_early`). Use
`require_budget_exhaustion` only for a deliberate budget-consumption ablation.

## Reproducible local execution

Install and verify the distribution with the pinned uv version:

```bash
cd environments/crafter/crafter
uv sync --frozen --extra dev
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check src scripts tests
uv run --frozen mypy src scripts tests
```

Example default LHS RGB Run:

```bash
uv run --frozen python scripts/run_crafter_codex.py \
  --model gpt-5.6-sol \
  --reasoning-effort xhigh \
  --record-to runs/crafter-sol-lhs-rgb \
  --profile lhs \
  --observation-profile rgb \
  --episode-budget 1024 \
  --max-episodes-per-submission 64 \
  --validation-episodes-per-candidate 256 \
  --assessment-episodes 512 \
  --seed 0 \
  --allow-unsafe-process
```

Use `--profile canonical` for native Crafter scoring and
`--observation-profile local-symbolic-v1` for the symbolic ablation.

The launcher's analysis environment includes NumPy for both observations and
Pillow for RGB image work. Agent instructions use `python` directly inside the
Run workspace because that workspace is not the uv project root.

`ProcessExecution.unsafe()` is process execution, not an operating-system
sandbox. `--allow-unsafe-process` records explicit acknowledgement. Stronger
isolation must be supplied by the caller around the entire Run, for example a
container or VM with controlled filesystem, process, and network access.

## Upstream and license

- Environment: `CrafterReward-v1`
- Upstream: `danijar/crafter==1.8.3`
- Upstream license: MIT
- Observation: RGB or separately identified local-symbolic v1
- Actions: 17 discrete actions
- Default horizon: 10,000 steps
