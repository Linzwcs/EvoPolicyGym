---
locale: en
page: crafter-policy-evolution
title: "Perception or Planning? Policy Evolution in Crafter"
description: "A paired RGB and local-symbolic experiment reveals different perception and long-horizon control bottlenecks in Crafter Policy evolution."
lead: "GPT-5.6 Sol, Terra, and Luna evolve markedly different Crafter Policies when local visual recognition is separated from long-horizon survival and development."
publishedAt: "2026-08-11"
date: "2026-08-11"
authors: [evopolicygym]
tags:
  - Benchmark
  - Crafter
  - Experiment
  - Policy Evolution
status: published
---

Crafter is an open-world survival game where an agent must stay alive while
exploring, gathering resources, fighting enemies, and progressing through a
crafting technology tree. Unlike environments with a short and well-defined
objective, success in Crafter requires many decisions to remain coordinated
over hundreds of steps.

For a coding Agent evolving a Policy, this creates two intertwined challenges:

1. **Perception:** recover nearby terrain, entities, inventory, and player
   status from the observation.
2. **Long-horizon control:** turn that state into a coherent strategy for
   survival, exploration, combat, and development.

In this experiment, we separate these two sources of difficulty. We evaluate
GPT-5.6 Sol, Terra, and Luna under the same Crafter task, but give their evolved
Policies either the original RGB observation or a local symbolic representation
of the same visible state.

The difference is substantial for **Sol and Terra**, but much smaller for
**Luna**. Removing most visual recognition work therefore exposes different
bottlenecks in the Policies evolved by the three coding Agents.

<!-- truncate -->

## Crafter as a Policy-Evolution Environment

Crafter procedurally generates a fresh world for every Episode. The player
begins without tools or resources and must balance immediate survival with
longer-term development.

A capable Policy needs to coordinate several behaviors:

- maintain food, drink, and health;
- explore an initially unknown world;
- collect increasingly advanced resources;
- craft and place tools and structures;
- avoid or fight hostile creatures;
- retain enough local information to revisit useful areas.

These objectives compete with one another. Exploration creates opportunities
for development but also exposes the player to danger. Crafting requires
resources that may be far from safety. A Policy that focuses only on immediate
survival can stagnate, while aggressive progression can quickly collapse if
basic needs are neglected.

Crafter therefore tests whether a coding Agent can evolve a **coordinated
long-horizon program**, rather than merely discover a locally useful action
rule.

## RGB vs. Local Symbolic Observations

The two conditions use the same simulator, procedural worlds, action space,
reward metric, Episode pools, and evaluation setup. Only the observation
exposed to the Policy changes.

### RGB

The RGB Policy receives the rendered `64 × 64 × 3` Crafter frame.

It must infer:

- terrain from colors and textures;
- creatures and objects from sprites;
- inventory and vitals from the HUD;
- player position and orientation from rendering;
- useful state under changing illumination, including nighttime.

### Local symbolic

The symbolic Policy instead receives structured information for the **same
local region**:

- local terrain and entity IDs;
- health, food, drink, energy, resources, and tools;
- facing direction, sleeping state, and daylight.

This representation removes most object recognition, HUD reading, and
nighttime visual ambiguity.

However, it does **not** expose privileged global state. The Policy still
receives no global semantic map, absolute position, Environment seed, hidden
creature state, or other information outside the local observation.

It must still explore, remember useful locations, handle collisions, sequence
resources, time interactions, fight enemies, and coordinate survival with
development.

The paired experiment therefore simplifies **state recognition** while
preserving most of the **long-horizon decision-making problem**.

## Long-Horizon Survival Score

Crafter's canonical score is primarily achievement-oriented. For our
policy-evolution setting, we additionally want to distinguish a Policy that
occasionally reaches advanced achievements from one that can survive reliably
while continuing to make progress.

We therefore use the **Long-Horizon Survival Score (LHS Score)** as the primary
Benchmark metric.

At each step, its survival component contains two signals:

- an **alive reward** for remaining alive;
- a **vital-quality reward** determined by the weakest of health, food, and
  drink.

Using the weakest vital is intentional: high food and health should not
compensate for a Policy that is about to die from thirst.

LHS also retains bounded secondary incentives for useful development:

- the first unlock of a new achievement;
- actual restoration of food or drink;
- productive repeated behaviors such as resource collection, combat, and
  planting.

Repeated actions are capped within rolling windows, preventing a simple farming
or maintenance loop from dominating the score.

Across Episodes, LHS further emphasizes robustness. The aggregate score
combines:

- average healthy-survival return;
- additional weight on the **weakest quarter of Episodes**;
- bounded development and maintenance return.

Conceptually,

`LHS = average survival + lower-tail robustness + bounded progression`

rather than rewarding only the best trajectories.

This makes short achievement-rich Episodes insufficient to compensate for a
fragile survival Policy, while still encouraging the Agent to progress beyond
passive survival.

The canonical Crafter score is reported separately as a diagnostic of
technology-tree progression.

## Experiment

We evaluate GPT-5.6 **Sol**, **Terra**, and **Luna**.

For each coding Agent, we run one policy-evolution trajectory with RGB
observations and one with local symbolic observations. All six Runs use the
same long-horizon survival objective.

The evolved candidate selected at the end of each Run is evaluated on **64
held-out Episodes**.

In addition to LHS, we report:

- mean and maximum effective survival;
- the fraction of Episodes surviving at least 300 steps;
- canonical Crafter score;
- achievement coverage.

These diagnostics allow us to distinguish average robustness, exceptionally
long trajectories, and technology progression.

## Results

The effect of simplifying perception differs sharply across the three Agents.

| Agent | LHS Score | Mean Survival | Max Survival | Survive ≥300 | Crafter C | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Sol** | 5.70 → **11.53** | 195 → **316** | 401 → **1043** | 3.1% → **39.1%** | 3.91 → **19.47** | 10 → **18/22** |
| **Terra** | 4.58 → **9.88** | 164 → **291** | 288 → **871** | 0.0% → **31.2%** | 1.12 → **11.31** | 5 → **14/22** |
| **Luna** | 4.58 → **4.97** | 164 → **178** | 288 → **401** | 0.0% → **3.1%** | 1.12 → **3.44** | 5 → **11/22** |

*Each cell compares RGB → local symbolic observations.*

For **Sol**, LHS increases from 5.70 to 11.53, while mean survival rises from
195 to 316 steps. Its longest held-out trajectory grows from 401 to **1,043
steps**.

**Terra** shows an equally pronounced shift. LHS more than doubles, mean
survival increases by 127 steps, and its longest trajectory reaches **871
steps**.

The change for **Luna** is much smaller. Symbolic observations increase
achievement coverage substantially, from 5 to 11 achievements, but mean
survival rises by only 13 steps and LHS improves by only 8%.

The difference is especially visible in robust long-horizon survival. Under
symbolic observations, **39.1%** of Sol Episodes and **31.2%** of Terra
Episodes survive at least 300 steps, compared with only **3.1%** for Luna.

For Sol and Terra, simplifying perception therefore changes not only their
best-case behavior, but the broader survival distribution.

*Terra's symbolic Run includes one validation protocol failure and one protocol
error during the held-out Assessment. The failed held-out Episode receives zero
under the Benchmark definition; among successfully completed Episodes, its
shortest survival is 156 steps.*

## One World, Six Policies

Aggregate results measure robustness across procedural worlds. To make the
behavioral differences easier to inspect, we also replay the six selected
Policies in the same showcase world.

The earlier fixed world happened to favor Luna. We replaced it using a
deterministic, separate 128-Episode showcase pool: eligible Episodes had to
complete normally, place Sol above the shared RGB baseline, and order the
symbolic Policies as Sol > Terra > Luna by effective survival, with every
adjacent gap at least 50 steps. Among eligible Episodes, we selected the one
closest to the corresponding held-out mean survival values,
rather than the one with the largest gap. Because this selection uses Policy
outcomes, the replay remains a **qualitative illustration**, not evaluation
evidence.

For the symbolic conditions, each Policy still receives only its structured
local observation. The RGB GIF is a human-facing deterministic replay of that
Policy's recorded Actions in the identical world; those RGB frames are not
Policy inputs. All six GIFs share the same timeline. A Policy that ends early
holds on its terminal frame.

| Sol | Terra | Luna |
| --- | --- | --- |
| **RGB · 244 steps**<br />![Sol RGB Policy on the shared Crafter showcase Episode](/images/blog/crafter-lhs-sol-rgb-showcase.gif) | **RGB · 162 steps**<br />![Terra RGB Policy on the shared Crafter showcase Episode](/images/blog/crafter-lhs-terra-rgb-showcase.gif) | **RGB · 162 steps**<br />![Luna RGB Policy on the shared Crafter showcase Episode](/images/blog/crafter-lhs-luna-rgb-showcase.gif) |
| **Symbolic · 391 steps**<br />![Sol local-symbolic Policy on the shared Crafter showcase Episode](/images/blog/crafter-lhs-sol-symbolic-showcase.gif) | **Symbolic · 261 steps**<br />![Terra local-symbolic Policy on the shared Crafter showcase Episode](/images/blog/crafter-lhs-terra-symbolic-showcase.gif) | **Symbolic · 194 steps**<br />![Luna local-symbolic Policy on the shared Crafter showcase Episode](/images/blog/crafter-lhs-luna-symbolic-showcase.gif) |

The symbolic row now reflects the aggregate ordering clearly: Sol continues
longest, Terra occupies the middle, and Luna ends much earlier. Terra RGB and
Luna RGB cannot be separated on a common Episode because both Runs selected
the byte-identical packaged baseline; their matching replays and 162-step
outcomes are intentional.

## Three Agents, Different Bottlenecks

The paired experiment produces three qualitatively different policy-evolution
trajectories.

### Sol: strong RGB progress, then another large gain

Sol already evolves a useful Policy from RGB observations. Its RGB result
clearly exceeds the packaged baseline, indicating that the Agent can make
progress while jointly solving visual interpretation and control.

Symbolic observations nevertheless produce another major jump.

Its selected symbolic Policy achieves:

- **11.53** LHS Score;
- **316** mean survival steps;
- **1,043** maximum survival steps;
- **18/22** achievement coverage;
- **19.47** canonical Crafter score.

Sol therefore appears capable of handling both parts of the problem, while
still benefiting strongly when state recognition becomes more reliable.

### Terra: perception was a major bottleneck

Terra follows a different trajectory.

Under RGB observations, its policy-evolution Run ultimately selects the
unmodified packaged baseline: its attempted visual Policies do not validate
above that fallback.

Once the same local world state is presented symbolically, the result changes
dramatically.

Terra reaches an LHS Score of **9.88**, mean survival of **291 steps**, maximum
survival of **871 steps**, and **14/22** achievement coverage.

The RGB result alone therefore substantially understates what Terra can do
once reliable local state is available.

For this Run, visual state extraction appears to have been a major bottleneck
upstream of long-horizon strategy.

### Luna: better recognition does not solve coordination

Luna also returns to the packaged baseline under RGB observations.

Symbolic observations help its Policy reach a substantially broader portion
of the technology tree: achievement coverage increases from **5/22 to 11/22**.

However, that development gain does not translate into comparable survival
robustness.

Mean survival increases only from 164 to 178 steps, maximum survival reaches
401 steps, and LHS moves from 4.58 to 4.97.

This suggests a different failure boundary from Terra.

For Luna, removing most of the perception problem exposes continuing
difficulty in coordinating survival, exploration, resource progression, and
action control over long horizons.

The three Agents are therefore not simply producing stronger or weaker
versions of the same strategy.

**Changing the observation contract affects their policy evolution in
fundamentally different ways.**

## What Does the Ablation Tell Us?

At first glance, Crafter poses a single challenge: evolve a Policy that can
survive and develop in an open world.

The paired experiment shows that the difficulty can originate at different
stages.

For **Sol and Terra**, local visual state extraction is a major part of the
challenge. Removing object recognition, HUD reading, and nighttime visual
ambiguity leads to much stronger survival and technology progression.

For **Luna**, perception is only part of the problem. Cleaner state information
enables broader development, but the resulting Policy still struggles to turn
those capabilities into reliable long-horizon survival.

This is the distinction we want environments such as Crafter to expose.

A final scalar score tells us which Program performs better. Controlled
changes to the observation interface can additionally reveal **where policy
evolution stops improving**.

Crafter therefore evaluates more than whether a coding Agent can write a
game-playing Policy. It gives us a way to separate failures of perception from
failures of long-horizon planning, control, and program coordination.

These six Runs represent individual policy-evolution trajectories rather than
repeated statistical trials, and their training Episode consumption is not
identical. We therefore do not interpret them as a general ranking of Sol,
Terra, and Luna.

The narrower observation is more informative:

> **Removing local visual recognition transforms the Policies evolved by Sol
> and Terra, but only modestly improves Luna's survival—revealing different
> bottlenecks behind long-horizon policy evolution.**
