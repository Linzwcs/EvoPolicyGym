# Crafter Long-Horizon Survival Score (LHS)

LHS is the default reward metric in the Crafter Benchmark distribution. Its
launcher name is `lhs`, its Benchmark class is
`CrafterLongHorizonSurvivalBenchmark`, and its primary Feedback field is
`long_horizon_survival_score`.

The metric distinguishes robust, healthy survival from short
achievement-rich trajectories while retaining bounded incentives for
development and maintenance. Native Crafter achievement scoring remains
available separately through the `canonical` profile.

## Transition reward

After Crafter applies Action `t`:

```text
alive_t = 1 unless t is the natural terminal transition, else 0
q_t = min(health_t, food_t, drink_t) / 9

alive_survival_t = 0.01 * alive_t
vital_survival_t = 0.03 * alive_t * q_t

Step.reward_t =
    alive_survival_t
    + vital_survival_t
    + first_unlock_delta_t
    + maintenance_repeat_delta_t
    + productivity_repeat_delta_t
```

At full visible vitals, one alive transition is worth `0.04`, 25 transitions
are worth one point, and a fully healthy 300-step day is worth 12 survival
points. The explicit alive component preserves a recovery incentive when one
necessity reaches zero. Energy and native Crafter reward are public diagnostics
and are not scored by LHS.

The named survival parameters are:

```text
LHS_ALIVE_ALPHA = 0.01
LHS_VITAL_ALPHA = 0.03
LHS_HEALTHY_WINDOW_STEPS = 300
```

## Development and maintenance

First unlocks use compressed technology-stage weights:

```text
first_unlock_credit(name) = 0.10 * log2(1 + raw_weight(name))
```

Productive repeats use a rolling 300-step window. Within one window, an event
can receive at most 20% of its own first-unlock credit, distributed across its
event-specific quota. The credited events are wood, saplings, stone, coal,
iron, diamond, zombies, skeletons, and planted crops.

Maintenance repeats have a total rolling-window allowance of 1.2 points,
split equally between drink and food. Credit is limited by the restoration
that was possible from the pre-Action value and by rolling restoration-unit
caps. Drinking and eating while already full therefore do not receive
maintenance credit.

An achievement's first successful event receives first-unlock credit only; it
does not also receive repeat credit.

## Episode return and Policy failure

A completed Episode return is the exact sum of public transition rewards:

```text
survival_return_i = sum(alive_survival + vital_survival)

secondary_return_i = sum(
    first_unlock
    + maintenance_repeat
    + productivity_repeat
)

episode_return_i = survival_return_i + secondary_return_i
```

A Policy execution failure has formal Episode return, survival return, and
secondary return zero. Its partial trajectory and partial component totals
remain visible and are explicitly marked as discarded.

## Cross-Episode Feedback

For `N` Episodes, let `k = max(1, ceil(0.25 * N))`. Episodes are sorted only by
`survival_return_i` when selecting the weak tail:

```text
LHS =
    0.75 * mean(survival_return)
    + 0.25 * mean(bottom-k survival returns)
    + mean(secondary_return)
```

There is no upper-tail bonus. Secondary return uses an ordinary mean and
cannot move a short, achievement-rich Episode out of the survival lower tail.
The named component contributions in Feedback exactly reconstruct the scalar
score.

## Public diagnostics

LHS Feedback publishes:

- mean, median, P10, P25, P90, minimum, and maximum effective survival;
- survival rates at 150, 200, 250, 300, and 400 steps;
- weakest-vital quality overall and by Episode-age bands;
- terminal food/drink status and health-change diagnostics;
- native Crafter return and canonical Crafter achievement score;
- first-unlock, maintenance, productivity, action, and Policy-failure details.

Every detailed training trajectory records the five LHS components, native
reward, health/food/drink values, and cadence diagnostics. Validation and test
publish aggregate Feedback only.

## Public identity

```text
RGB Benchmark ID:
crafter/CrafterReward-v1/long-horizon-survival-score-v1

Local-symbolic Benchmark ID:
crafter/CrafterReward-v1/local-symbolic-v1/long-horizon-survival-score-v1

reward_profile: lhs
primary_metric: long_horizon_survival_score
```
