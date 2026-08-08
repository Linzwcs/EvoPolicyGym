---
locale: en
page: crafter-policy-evolution
title: "Survive, Build, Repeat: Shaping Policy Evolution in Crafter"
description: "How achievement, survival, and repeat-production Feedback redirected executable Policy evolution in Crafter."
lead: "A reward ablation with GPT-5.6 Luna, Terra, and Sol shows the tension between surviving longer and progressing through Crafter's technology tree."
publishedAt: "2026-08-09"
date: "2026-08-09"
authors: [evopolicygym]
tags:
  - Benchmark
  - Crafter
  - Experiment
  - Policy Evolution
status: published
---

## What is Crafter?

Crafter is an open-world survival game built around exploration, resource
gathering, crafting, combat, and survival. Each Episode takes place in a
randomly generated world. The player must find food and water, avoid or fight
creatures, collect materials, and progress from wood and stone to iron and
diamond.

<!-- truncate -->

A simplified progression loop looks like this:

```text
explore the world
        ↓
gather food, water, and materials
        ↓
survive long enough to craft tools
        ↓
unlock new resources and abilities
        ↓
reach harder achievements
```

Crafter evaluates 22 meaningful achievements spanning resources, creatures,
tools, farming, and construction. Its standard score takes a shifted geometric
mean of achievement success rates. This rewards broad progress across the
technology tree rather than repeatedly completing one easy task.

That creates a strategy problem: progress requires survival, but optimizing
only for survival can produce conservative behavior that never reaches the
deeper technology tree.

## Bringing Crafter into EvoPolicyGym

The Benchmark uses Crafter 1.8.3 with 10,000-step Episodes. The Policy receives
only a `64 × 64 × 3` RGB observation and chooses from 17 discrete Actions.
Health, food, water, and inventory are visible only through the rendered
screen; player position, semantic maps, and Environment seeds are not exposed
as structured Policy inputs.

The standard Crafter score measures achievement breadth well, but it does not
directly distinguish a sustainable long-lived routine from a brief burst of
progress before death. We therefore evaluated four aggregate Feedback
profiles:

| Profile | Feedback score | Purpose |
| --- | --- | --- |
| **M1 · Achievement** | `C` | Standard Crafter achievement progress |
| **M2 · Survival** | `C + L / 100` | Add a light survival incentive |
| **M3 · Survival + Repeat** | `C + L / 100 + R` | Balance progress, survival, and repeatable production |
| **M4 · Strong Survival + Repeat** | `C + L / 20 + R` | Strongly prioritize survival |

`C` is the standard Crafter achievement score, `L` is mean effective survival
length, and `R` rewards confirmed repeated activities such as gathering,
drinking, eating, building, and defeating creatures. The first successful
event remains part of `C`; only later successes contribute to `R`. Repeat
credit grows logarithmically and has a continuous survival-scaled limit, so a
short-lived Policy cannot obtain an arbitrarily large score from one easy
loop.

These profiles do **not** change Crafter's original step reward. They change
only the aggregate Feedback used to compare submitted Programs.

## Learning from visual trajectories

Aggregate scores reveal whether a Policy improved, but many Crafter failures
are easier to understand visually. A Policy may circle within one area, ignore
food or water, remain limited to low-level resources, or discover a productive
loop without progressing further.

For every training submission, the Agent receives the complete Policy-visible
Action trajectory and lossless RGB observations. Validation and held-out test
trajectories remain private. NumPy, Pillow, and image-viewing tools let the
coding Agent inspect what the Policy saw and did before rewriting the
executable strategy. The Environment supplies evidence; choosing what to
inspect and how to interpret it remains part of the Agent's work.

## Experiment

We ran GPT-5.6 Luna, Terra, and Sol through Codex. Each Agent started from the
same packaged baseline and optimized it independently under each reward
profile. All 12 Runs shared the same Environment, split construction, and Run
seed; within each model lane, the reward profile was the experimental
variable. The optional Benchmark Skill was disabled.

| Setting | Value |
| --- | --- |
| Environment | Crafter 1.8.3 |
| Policy input | `64 × 64 × 3` RGB |
| Actions | 17 |
| Episode horizon | 10,000 steps |
| Training allowance | up to 256 Episodes |
| Maximum per submission | 16 Episodes |
| Validation | 32 Episodes per candidate |
| Final Assessment | 64 held-out Episodes |
| Agent harness | Codex, high reasoning effort |
| Optional Benchmark Skill | disabled |

The training allowance was a ceiling, not forced consumption, so Agents could
finish before using all 256 Episodes.

On the fixed held-out pool, the baseline had a Crafter achievement score of
`1.025`, survived for `163.4` effective steps on average, and reached only
`5/22` achievement types. Its behavior was dominated by collecting wood,
drinking water, and collecting saplings.

## Results

All three Agents improved over the packaged baseline on the aggregate objective
used in their Runs, but the resulting Policies differed substantially across
both Agents and objectives.

| Metric | Agent | Final score | Achievement `C` | Survival `L` | Repeat `R` | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **M1 · Achievement** | **Sol** | **12.748** | **12.748** | 183.8 | — | **17/22** |
|  | Luna | 3.461 | 3.461 | 163.3 | — | 11/22 |
|  | Terra | 1.655 | 1.655 | 176.8 | — | 7/22 |
| **M2 · + Survival** | **Sol** | **10.110** | **8.308** | **180.2** | — | **15/22** |
|  | Luna | 3.563 | 1.803 | 176.0 | — | 8/22 |
|  | Terra | 3.918 | 2.134 | 178.5 | — | 7/22 |
| **M3 · + Survival + Repeat** | Sol | 7.073 | 4.007 | 177.8 | 1.288 | 10/22 |
|  | Luna | 6.651 | 2.305 | 177.7 | **2.570** | 9/22 |
|  | **Terra** | **7.997** | **4.092** | **186.8** | 2.037 | **11/22** |
| **M4 · Strong Survival + Repeat** | **Sol** | **15.501** | **4.717** | 176.5 | 1.957 | **14/22** |
|  | Luna | 14.193 | 2.253 | **185.9** | 2.647 | 10/22 |
|  | Terra | 13.055 | 1.023 | 184.7 | **2.799** | 5/22 |

Final scores should be compared with the corresponding profile's baseline,
not directly across profiles: M4 deliberately assigns more points to every
survival step than M1–M3.

M1 produced the broadest technology-tree progression. Sol reached `17/22`
achievements and a standard Crafter score of **12.748**, compared with `11/22`
for Luna and `7/22` for Terra. M2 increased survival for all three Agents while
Sol retained substantially broader achievement progress.

M3 produced the most balanced result. Within that profile, Terra achieved the
highest final score at **7.997**, combining `C = 4.092`, `186.8` effective
survival steps, and non-zero success on `11/22` achievements. Its repeat credit
came from a mixture of wood and stone collection, placing stone, eating cows,
and drinking rather than one maintenance action alone.

Across the three M4 Runs, mean survival increased, but the Policies did not all
progress broadly. Sol still reached `14/22` achievements, whereas Terra
remained at `5/22`, the same coverage as the baseline. Terra's standard
achievement score, `1.023`, was also slightly below the baseline despite its
much higher aggregate M4 score.

Across Agents, increasing the survival incentive produced a consistent
aggregate trade-off:

| Metric | Mean achievement `C` | Mean survival `L` | Mean repeat `R` |
| --- | ---: | ---: | ---: |
| M1 | **5.955** | 174.6 | — |
| M2 | 4.082 | 178.2 | — |
| M3 | 3.468 | 180.8 | 1.965 |
| M4 | 2.664 | **182.4** | **2.468** |

Mean survival rose monotonically from `174.6` to `182.4` steps, while the mean
standard Crafter score fell from `5.955` to `2.664`. Reward shaping therefore
redirected Policy evolution rather than merely rescaling the same behavior.

![Step-aligned, side-by-side sampled replays of representative M3 training
Episodes from the Validation-selected Sol, Luna, and Terra Policies. The three
Policies explore, gather, craft, and survive in different generated
worlds.](/images/blog/crafter-m3-policy-replay-comparison.gif)

*Representative, non-seed-matched M3 training trajectories. Sol uses train
index 159 from Submission 000010 and reaches 7 achievements in 296 steps; Luna
uses index 62 from Submission 000004 and reaches 7 achievements in 194 steps;
Terra uses index 120 from Submission 000008 and reaches 8 achievements in 294
steps. The replay displays one frame every three Policy steps on a shared step
timeline; after an Episode ends, its final observation remains visible. These
Episodes are not part of the held-out Assessment.*

## The baseline's capability boundary

The baseline can answer an early-game question:

> How do I obtain nearby resources and satisfy immediate survival needs?

The broader task is different:

> How should I coordinate survival, exploration, production, and the
> technology tree over a long Episode?

Its `5/22` achievement coverage shows that it can enter the early game but
rarely develops a broad progression chain. Policy evolution must decide when
to gather familiar resources, when to explore, when to invest in tools and
structures, and how much effort to spend simply staying alive.

## How reward shaping changed Policy evolution

**M1 favored technology-tree breadth.** Without an explicit survival bonus,
new achievement types remained the dominant route to a higher score.

**M2 added survival without substantially redefining the task.** Mean survival
improved by about 15 steps over the baseline while achievement score remained
above baseline for all three Agents.

**M3 rewarded sustainable progression.** Repeated production, food, combat,
and construction contributed after their first achievement, while the light
survival term discouraged immediate death without overwhelming the original
objective.

**M4 pushed the trade-off too far.** It gained only about `1.6` mean survival
steps over M3 while losing roughly `0.8` mean achievement points. Drinking
alone contributed about 65% of Sol's repeat score, 86% of Luna's, and 83% of
Terra's. The stronger survival term made low-risk maintenance loops
disproportionately attractive.

This is a useful failure mode for an executable-Policy Benchmark: a coding
Agent can discover and encode behavior that exploits whichever incentives the
Environment makes durable.

## Findings and boundaries

Among these four profiles, **M3 is the most useful long-survival variant**. It
captures most of M4's survival improvement while preserving more achievement
progress and supporting a broader mix of repeatable activities. M1 remains the
standard Crafter control, M2 a simple survival hybrid, and M4 a strong-survival
ablation.

This is an initial reward ablation. Each model/profile pair is represented by
one primary coding-agent trajectory, and the Agents did not consume equal
training budgets. The results show that reward design systematically redirected
Policy evolution in these Runs; they are not a general ranking of Luna, Terra,
and Sol.

More broadly, Crafter exposes a useful tension for EvoPolicyGym: **a good
metric must reward staying alive long enough to build a strategy without
making survival itself easier to optimize than progress.**

## Code and notes

- [Crafter upstream Environment](https://github.com/danijar/crafter)
- [EvoPolicyGym Crafter Benchmark](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/crafter/crafter)
- [Evaluation and Runs](/docs/evaluation/)
- [Policy boundary](/docs/policy/)

The EvoPolicyGym adapter is MIT licensed. Crafter remains a separate dependency
governed by its own license.
