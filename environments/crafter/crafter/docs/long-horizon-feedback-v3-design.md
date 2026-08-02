# Crafter long-horizon Feedback v3 design proposal

Status: proposal for review; no runtime behavior implements this document yet.

This document specifies only the scalar `Feedback.score` and the
Benchmark-defined public `Feedback.content`/trajectory scoring fields for a new
Crafter long-horizon profile. It deliberately does not prescribe submission
batch sizes, train budgets, Validation Episode counts, Assessment Episode
counts, or Agent search strategy.

## 1. Scope and compatibility

The proposed profile is additive and NLE-inspired, but survival remains the
primary Crafter objective. Its tentative identity is:

```text
crafter/CrafterReward-v1/mean-survival-development-return-v3
```

The canonical achievement profile and
`crafter/CrafterReward-v1/long-horizon-development-v2` remain unchanged. v3 is
a new Benchmark identity; v1, v2, and v3 scores are not numerically comparable.

The proposal does not change:

- the `64 x 64 x 3` uint8 RGB Policy observation;
- the 17-Action Policy ABI;
- the hidden, split-scoped Episode seeds;
- the fresh Policy process and instance per Episode;
- the lossless RGB observation Artifacts;
- temporal bulk retention;
- Agent-owned analysis authority;
- Host-only Validation and Assessment evidence boundaries.

The scorer may use trusted Crafter inventory counters that correspond to HUD
information already rendered into the RGB observation. These values are not
added to the live Policy observation.

For this new profile, the recommended Environment `Step.reward` is the shaped
scoring delta defined below. The pinned Crafter 1.8.3 reward is retained as a
separate diagnostic field. This makes Environment Episode return, trajectory
reward, and the successful-Episode score agree, as they do in NLE. Existing
v1/v2 profiles keep their current reward meaning unchanged.

## 2. Design goals

The score should satisfy these properties:

1. Every additional step survived has positive value until the 10,000-step
   horizon. There are no 300/600/900 marginal-value discontinuities.
2. Survival is the largest practical source of return. Maintenance and
   development refine a surviving Policy rather than replacing survival.
3. Maintaining vitals before an emergency is valuable. The score must not
   require a vital to cross a warning threshold before recovery can count.
4. First-time capability progress and repeated useful production both count,
   but repeated events have bounded diminishing returns.
5. The scalar score is the exact sum of published additive components.
6. A death stops future return but does not erase legitimate prior progress.
7. Only precisely defined game outcomes affect the score. Action cycles and
   visual heuristics remain diagnostics rather than penalties.
8. Feedback covers every evaluated Episode and does not select evidence by
   score, duration, achievement, or death cause.

## 3. Per-Episode score

### 3.1 Successful Episode

For transition `t`, define:

```text
alive_t = 0 if the transition naturally terminates the Episode, otherwise 1
```

A horizon-truncated final transition has `alive_t = 1`: the player survived
that transition. This preserves the current effective-survival convention:

```text
survival_credit = sum(alive_t)
                = steps - int(naturally_terminated)
```

Let the post-Action trusted vitals be health `h_t`, food `f_t`, drink `d_t`,
and energy `e_t`, each in `[0, 9]`. Define continuous vital quality:

```text
vital_quality_t = min(h_t, f_t, d_t, e_t) / 9
vital_credit_t  = 0.10 * alive_t * vital_quality_t
```

Using the minimum makes the weakest survival subsystem visible without a hard
warning threshold. The maximum maintenance contribution is ten percent of
survival credit.

Let `I_t` be the normalized first-unlock progress potential and `P_t` the
normalized repeated-productivity potential after transition `t`. Both are in
`[0, 1]` and never decrease. Define:

```text
progress_potential_t     = 75 * I_t
productivity_potential_t = 25 * P_t

progress_delta_t = progress_potential_t - progress_potential_(t-1)
productivity_delta_t = (
    productivity_potential_t - productivity_potential_(t-1)
)
```

The shaped scoring delta is:

```text
score_delta_t = (
    alive_t
    + vital_credit_t
    + progress_delta_t
    + productivity_delta_t
)
```

For the v3 profile:

```text
Step.reward                       = score_delta_t
Step.metrics["upstream_reward"] = pinned Crafter 1.8.3 reward
```

The Episode return is exactly:

```text
episode_return = sum(score_delta_t)
               = survival_credit
               + vital_credit
               + progress_credit
               + productivity_credit
```

Consequences of this scale:

- vital credit is at most `0.10 * survival_credit`;
- first-unlock progress is at most 75 points per Episode;
- repeated productivity is at most 25 points per Episode;
- all non-survival development credit is bounded at 100 points;
- survival remains valuable after step 900 and up to the configured horizon.

There is no separate death penalty. Death already removes the terminal
transition's alive/vital credit and prevents every future transition from
earning return.

### 3.2 Policy failure

Following the NLE failure contract, any Policy failure discards partial credit:

```text
episode_return = -max_episode_steps
```

The Episode component record is:

```text
survival_credit     = 0
vital_credit        = 0
progress_credit     = 0
productivity_credit = 0
failure_adjustment  = -max_episode_steps
```

Environment or evaluator faults remain trusted failures and must never be
converted into this Policy penalty.

### 3.3 Batch score

The scalar Feedback score is the ordinary arithmetic mean:

```text
Feedback.score = mean(episode_return)
```

The proposed primary metric name is:

```text
mean_survival_development_return
```

## 4. First-unlock progress potential

For achievement `i`, let `u_i` be 1 after its first successful event in the
Episode and 0 before it. Each achievement receives a dependency-stage weight
`w_i`:

| Weight | Achievements |
|---:|---|
| 1 | `collect_drink`, `collect_sapling`, `collect_wood`, `eat_cow`, `eat_plant`, `wake_up` |
| 2 | `place_table`, `place_plant`, `make_wood_pickaxe`, `make_wood_sword`, `defeat_zombie` |
| 3 | `collect_stone`, `place_stone`, `make_stone_pickaxe`, `make_stone_sword`, `defeat_skeleton` |
| 4 | `collect_coal`, `place_furnace` |
| 6 | `collect_iron`, `make_iron_pickaxe`, `make_iron_sword` |
| 8 | `collect_diamond` |

The weights sum to 65. The normalized potential is:

```text
I_t = sum(w_i * u_i) / 65
```

This preserves credit for all 22 canonical achievements while making later
dependency stages more valuable than repeatedly solving only the opening.
Weights describe capability depth, not empirical achievement rarity in one
particular run.

## 5. Repeated-productivity potential

For eligible event `i`, let `n_i` be its cumulative successful event count and
`r_i = max(n_i - 1, 0)` its repeat count after the first event. With cap `c_i`
and weight `v_i`, define:

```text
repeat_credit_i = log1p(min(r_i, c_i)) / log1p(c_i)
P_t = sum(v_i * repeat_credit_i) / sum(v_i)
```

Only upstream-confirmed successful events count. Returning an Action, a failed
craft, an unsuccessful attack, or `do` beside a nonproductive target does not
count.

The proposed repeat table is:

| Event | Weight | Repeat cap |
|---|---:|---:|
| `collect_drink` | 3 | 8 |
| `eat_cow` | 3 | 4 |
| `eat_plant` | 3 | 4 |
| `collect_wood` | 1 | 8 |
| `collect_sapling` | 1 | 4 |
| `collect_stone` | 2 | 8 |
| `collect_coal` | 3 | 4 |
| `collect_iron` | 4 | 3 |
| `collect_diamond` | 6 | 2 |
| `defeat_zombie` | 3 | 4 |
| `defeat_skeleton` | 4 | 2 |
| `place_plant` | 2 | 4 |
| `place_stone` | 2 | 8 |
| `place_table` | 1 | 2 |
| `place_furnace` | 2 | 2 |

The repeat weights sum to 40. Repeated tool and sword crafting is excluded:
tools do not wear out in pinned Crafter 1.8.3, so manufacturing redundant
copies consumes resources without increasing capability. Repeated drinking
and eating receive small bounded production credit even when not urgent;
timeliness is scored separately by the vital-quality integral.

## 6. What is deliberately not scored

The following remain public diagnostics only:

- immediate reverse movement;
- repeated short Action cycles;
- same-Action runs;
- movement percentage;
- `survival@300/600/900`;
- inferred death categories;
- canonical Crafter score;
- upstream Crafter environment return.

Crafter advances world time on every valid Action, so it has no direct analog
of NLE's frozen-turn penalty. A generic pixel-similarity or Action-cycle
penalty could punish legitimate combat, sleeping, waiting inside a shelter, or
repeated gathering and is therefore excluded from v3 scoring.

### 6.1 World-development coverage and limits

Pinned Crafter 1.8.3 has a smaller world-development vocabulary than a general
sandbox game:

- cultivation is represented by `place_plant` and `eat_plant`; a placed plant
  requires more than 300 nearby world updates to become ripe and can regrow
  after harvest;
- productive facilities are represented by placing a table/furnace and then
  successfully crafting tools that require those nearby facilities;
- construction is represented only by successful stone/table/furnace
  placement events; the upstream game exposes no house or enclosure state;
- animal breeding is not implemented. Cows are spawned and despawned by the
  environment balance process rather than reproduced by a player Action.

The proposed score already gives bounded credit to plant placement, ripe-plant
harvest, facility placement, dependent tool production, and repeated stone
placement. It does not add a separate house score. Inferring an enclosure from
trusted map topology would introduce a new private spatial classifier, while
rewarding only stone count would confuse a useful wall with arbitrary block
spam. Successful shelter construction is therefore rewarded indirectly by
the survival return rather than by a claimed house event.

Feedback should group the upstream-confirmed events for interpretation without
inventing new ground-truth states:

```json
{
  "world_development_diagnostics": {
    "cultivation": {
      "plants_placed": 0,
      "ripe_plant_harvests": 0
    },
    "facilities": {
      "tables_placed": 0,
      "furnaces_placed": 0,
      "dependent_tools_made": 0
    },
    "construction": {
      "stone_blocks_placed": 0,
      "enclosure_geometry_verified": false
    },
    "animal_breeding_supported": false,
    "scored_beyond_published_event_components": false
  }
}
```

These are grouped views of existing public event counters, not additional
score terms or private world-state disclosure. Whether cultivation needs more
weight should be decided from v3 rollouts rather than by adding an unbounded
building bonus before calibration.

## 7. Aggregate Feedback.content

The proposed compact content schema is:

```json
{
  "schema": "crafter/mean-survival-development-feedback/v3",
  "score_profile": "mean-survival-development-return-v3",
  "summary": "Mean survival-development return ...",
  "mean_survival_development_return": 0.0,
  "scoring_parameters": {
    "survival_credit_per_alive_step": 1.0,
    "vital_credit_scale": 0.1,
    "vital_quality_formula": "min(health, food, drink, energy) / 9",
    "progress_credit_max": 75.0,
    "productivity_credit_max": 25.0
  },
  "score_components": {
    "mean_survival_credit": 0.0,
    "mean_vital_credit": 0.0,
    "mean_progress_credit": 0.0,
    "mean_productivity_credit": 0.0,
    "mean_failure_adjustment": 0.0,
    "reconstructed_mean_return": 0.0,
    "reconstruction_error": 0.0,
    "formula": "mean(survival + vital + progress + productivity + failure)"
  },
  "episode_returns": {
    "mean": 0.0,
    "median": 0.0,
    "p10": 0.0,
    "p90": 0.0,
    "min": 0.0,
    "max": 0.0
  },
  "episodes": 0,
  "terminated_episodes": 0,
  "truncated_episodes": 0,
  "policy_failures": 0,
  "failure_return": -10000.0,
  "survival_steps": {
    "mean": 0.0,
    "median": 0.0,
    "p10": 0,
    "p90": 0,
    "min": 0,
    "max": 0
  },
  "survival_at_steps": {
    "300": {"count": 0, "percent": 0.0},
    "600": {"count": 0, "percent": 0.0},
    "900": {"count": 0, "percent": 0.0}
  },
  "vital_quality": {
    "mean": 0.0,
    "mean_min_vital": 0.0,
    "zero_min_vital_steps": 0,
    "zero_min_vital_step_fraction": 0.0,
    "steps_at_or_below": {
      "2": {
        "health": 0,
        "food": 0,
        "drink": 0,
        "energy": 0
      },
      "5": {
        "health": 0,
        "food": 0,
        "drink": 0,
        "energy": 0
      }
    }
  },
  "terminal_profile": {
    "natural_deaths": 0,
    "deaths_with_food_zero": 0,
    "deaths_with_drink_zero": 0,
    "deaths_with_food_and_drink_positive": 0,
    "terminal_vital_means": {
      "health": 0.0,
      "food": 0.0,
      "drink": 0.0,
      "energy": 0.0
    }
  },
  "health_change_diagnostics": {
    "loss_events": 0,
    "health_lost": 0,
    "recovery_events": 0,
    "health_recovered": 0,
    "loss_by_action": {
      "move_left": {"events": 0, "amount": 0, "terminal_events": 0}
    },
    "scored": false
  },
  "progress": {
    "maximum_credit_per_episode": 75.0,
    "weight_sum": 65,
    "achievement_weights": {},
    "achievement_success_percent": {},
    "achievement_mean_credit": {}
  },
  "productivity": {
    "maximum_credit_per_episode": 25.0,
    "weight_sum": 40,
    "repeats_only": true,
    "event_weights": {},
    "event_caps": {},
    "event_counts": {},
    "event_mean_credit": {},
    "event_saturation_percent": {}
  },
  "canonical_comparison": {
    "crafter_score_percent": 0.0,
    "mean_upstream_return": 0.0
  },
  "world_development_diagnostics": {
    "cultivation": {
      "plants_placed": 0,
      "ripe_plant_harvests": 0
    },
    "facilities": {
      "tables_placed": 0,
      "furnaces_placed": 0,
      "dependent_tools_made": 0
    },
    "construction": {
      "stone_blocks_placed": 0,
      "enclosure_geometry_verified": false
    },
    "animal_breeding_supported": false,
    "scored_beyond_published_event_components": false
  },
  "episode_score_summaries": [],
  "action_diagnostics": {},
  "detailed_feedback": {}
}
```

All configured weights and caps are published in content and
`BenchmarkSpec.metadata`; the score has no hidden coefficients.

`abs(reconstruction_error)` must be at most `1e-12`. A larger value is a
trusted scoring error, not ordinary Agent feedback.

`vital_quality` counts only transitions with `alive_t = 1`; terminal vitals
are reported separately. `terminal_vital_means` is computed only over natural
deaths and is `null` when there are no natural deaths.

`achievement_mean_credit[name]` and `event_mean_credit[name]` are arithmetic
means of that named event's realized credit across all evaluated Episodes,
including zero-credit Episodes. `event_saturation_percent[name]` is the
percentage of Episodes whose repeat count reached that event's published cap.
These fields partition the corresponding aggregate component and therefore
must reconstruct it within the same tolerance.

## 8. Per-Episode score summaries

Compact Feedback includes every Episode ordinal in evaluation order:

```json
{
  "episode_index": 0,
  "status": "completed",
  "terminated": true,
  "truncated": false,
  "failure": null,
  "steps": 180,
  "effective_survival_steps": 179,
  "return": 198.25,
  "components": {
    "survival_credit": 179.0,
    "vital_credit": 12.75,
    "progress_credit": 6.0,
    "productivity_credit": 0.5,
    "failure_adjustment": 0.0
  }
}
```

The list is complete and ordinal-only. It contains no seed, stable Case ID,
Host path, Policy seed, or private simulator state.

For a Policy failure, `status` is `policy_failed`, `return` equals the
published failure return, the four positive components are zero, and the
failure code remains the Kernel-owned public failure classification.

## 9. Trajectory scoring fields

The lossless RGB observation format and alignment remain unchanged. Because
v3 changes the meaning of Environment reward under a new benchmark identity,
its trajectory and manifest declare a new schema version. Each training
trajectory transition records:

```json
{
  "reward": 1.0444444444,
  "upstream_reward": -0.2,
  "score_delta_components": {
    "survival": 1.0,
    "vital": 0.0444444444,
    "progress": 0.0,
    "productivity": 0.0
  }
}
```

`reward` always means the actual `Step.reward` received from this v3
Environment. Existing v1/v2 Artifacts continue to use their existing upstream
reward meaning and are never reinterpreted. The explicit schema/profile
identity prevents the two formats from being mixed.

The trajectory does not publish raw trusted inventory values. The vital score
delta reveals only the normalized weakest-vital credit used by the scorer;
the exact four HUD values must still be decoded from the RGB observation.

For a Policy-failed Episode, retained transitions keep the shaped rewards that
were actually observed before failure, but the Episode header marks them
`scored: false`, reports `partial_return`, and records the scored result as
`scored_return = -max_episode_steps`. The compact Episode summary zeros the
four positive components and records the full failure adjustment. Partial
transition rewards therefore remain truthful debugging evidence but never
contribute to `Feedback.score`.

Validation and Assessment retain aggregate `Feedback.content` in Host reports
but publish no trajectory or RGB Artifacts to the Agent workspace, preserving
the existing phase boundary.

## 10. Diagnostic interpretation

The proposed feedback distinguishes these cases without changing the score:

- high survival plus high vital quality: stable maintenance;
- high survival plus low vital quality: fragile survival close to starvation,
  dehydration, exhaustion, or death;
- repeated drink/eat events but low vital quality: actions are mistimed or the
  controller neglects another vital;
- high progress but low productivity: one-time capability acquisition without
  sustained use;
- high productivity but low progress: repeated basic behavior without deeper
  development;
- deaths with positive food and drink plus repeated health loss: combat or
  environmental danger is more likely than pure deprivation;
- long Action cycles with acceptable survival: visible controller behavior,
  not automatically a scoring exploit.

Death causes remain explicitly described as diagnostic evidence, not ground
truth labels. Crafter does not provide one authoritative mutually exclusive
cause for every death.

## 11. Required implementation tests

Before enabling the profile, tests should cover:

1. every alive step adds exactly one survival point;
2. natural terminal transitions add no survival credit, while horizon
   truncation does;
3. vital credit is continuous, bounded, and uses exactly the finalized public
   vital-quality formula;
4. first-unlock weights cover exactly all 22 achievements and sum to 65;
5. repeat weights/caps cover exactly the published repeat table and sum to 40;
6. repeat credit excludes the first event and saturates at its cap;
7. failed Actions with no achievement-counter increment receive no progress or
   productivity delta;
8. component totals reconstruct every Episode return and the batch score;
9. Policy failure receives exactly `-max_episode_steps` and no partial credit;
10. environment faults are not converted into Policy penalties;
11. compact content and trajectory fields contain no seed, path, Case identity,
    raw inventory dictionary, or process evidence;
12. canonical comparison metrics remain unchanged;
13. Artifact count, per-Artifact size, retention class, and observation bytes
    remain within the existing contract;
14. in v3, `Step.reward` equals the shaped score delta and the pinned upstream
    reward is preserved separately;
15. failed-Episode partial transition rewards are retained as unscored
    evidence and never contribute to the batch score.

## 12. Review decisions

Review status as of 2026-08-02:

| Decision | Status |
|---|---|
| Raw energy in continuous vital quality | Pending after mechanics review |
| Use the weakest included vital (`min`) | Accepted |
| 75/25 first-progress/repeated-productivity split | Accepted for initial calibration |
| Dependency-stage weights and repeat table | Accepted for initial calibration |
| Publish per-transition score deltas | Accepted |
| Policy failure return is `-max_episode_steps` | Accepted |
| v3 `Step.reward` is the shaped delta; preserve `upstream_reward` separately | Accepted |

Energy requires one remaining decision. In pinned Crafter 1.8.3, awake energy
drops by one after 31 awake updates. Sleeping restores one energy after a
little over ten sleeping updates, halves hunger/thirst accumulation, and
doubles the health-recovery accumulator. Energy does not restrict movement,
collection, combat, placement, or crafting. Its direct life-system effect is
binary: while awake, `energy > 0` satisfies the energy necessity; while
sleeping, the necessity is satisfied even at zero energy.

For that reason, the updated recommendation is to exclude raw energy from the
continuous minimum and use:

```text
vital_quality_t = min(health_t, food_t, drink_t) / 9
```

Energy, sleeping, and wake-up counts would remain explicit diagnostics and
survival consequences would still affect the primary score. Including raw
energy instead would assign different maintenance value to energy 8 versus 9
even though the game gives them identical capabilities, and may over-reward
frequent sleep cycles. No implementation should begin until this final energy
choice is accepted or revised.

## 13. MineRL comparison and adaptation rationale

This section is informative: it records why the proposed Crafter design uses
its particular decomposition. The reference implementation reviewed was the
official MineRL v0.4 branch at commit
`ccc8c659d2b8e4737eda418593dee94c101c9ba3`.

MineRL does not use one universal Minecraft score. It chooses a reward contract
according to what the task can verify:

1. `Treechop` gives one point for every newly obtained log and ends after 64
   logs. The repeated event is the objective itself.
2. `Obtain` publishes both a sparse first-acquisition variant and a dense
   every-acquisition variant. Both use a prerequisite hierarchy whose rewards
   increase strongly toward diamond: log 1, planks 2, stick/table 4, wood
   pickaxe 8, cobblestone 16, furnace/stone pickaxe 32, iron ore 64, iron
   ingot 128, iron pickaxe 256, and diamond 1024.
3. The item handlers count positive inventory changes, distinguish first-only
   from repeated acquisition, and request loop exclusion from the simulator.
4. `NavigateDense` adds signed progress toward a known goal to the terminal
   goal reward. This works because the target and distance are unambiguous.
5. generic `Survival` defines no reward, while BASALT construction tasks such
   as building a village house also expose no automatic reward and are judged
   by humans. MineRL therefore does not treat block count as a trustworthy
   substitute for whether a structure is a good house.

The v3 Crafter adaptation deliberately combines, rather than copies, these
contracts:

- first-unlock progress corresponds to MineRL's first-acquisition hierarchy;
- repeated productivity corresponds to dense acquisition, but uses published
  caps and logarithmic diminishing returns because it is secondary to
  survival;
- survival provides the dominant dense signal because long-lived open-world
  development, rather than one terminal item, is this profile's objective;
- absolute MineRL-style exponential rewards are normalized into the fixed
  75-point progress budget, so a rare late milestone cannot numerically erase
  survival;
- `achievement_mean_credit`, `event_mean_credit`, and
  `event_saturation_percent` make the realized reward sources inspectable,
  rather than showing only a total;
- place/collect reuse is initially bounded by the 25-point productivity budget
  and per-event caps. Suspected reuse cycles remain diagnostics. A private map
  classifier is not introduced merely to emulate MineRL's loop exclusion;
- no automatic house, aesthetics, farm-layout, or breeding score is invented
  where pinned Crafter has no authoritative corresponding state.

This comparison supports keeping the initial 75/25 calibration. If rollouts
show that the bounded repeated component is still dominated by resource reuse,
the next profile should change the event eligibility contract explicitly; it
must not silently reinterpret v3 results.
