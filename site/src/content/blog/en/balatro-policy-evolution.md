---
locale: en
page: balatro-policy-evolution
title: "Letting Coding Agents Build Strategy Systems for Balatro"
description: "We integrated a Balatro Environment and compared the policy-optimization results of Luna, Terra, and Sol under a 1,024-Episode interaction budget."
lead: ""
publishedAt: "2026-07-29"
author: "EvoPolicyGym contributors"
tags:
  - Benchmark
  - Balatro
  - Experiment
  - Policy Evolution
status: draft
---

## What is Balatro?

Balatro is a roguelike deck-building game centered on scoring poker hands. A
player begins each Run with a standard deck, scores points by playing hands,
strengthens the build through the shop, and wins by defeating the Boss Blind
in Ante 8. Draws, shops, and rewards change from Run to Run; losing starts a
new Run.

Each Run repeatedly follows the same loop:

```text
choose a Blind → play or discard → reach the target score → earn money → build in the shop → next Blind
```

- **Blinds**: A Run has eight Antes, each containing a Small Blind, Big Blind,
  and Boss Blind. The Small and Big Blinds can be skipped in exchange for a
  Tag, while the Boss Blind adds a rule that changes how the round plays.
- **Playing a hand**: The player selects one to five cards. Poker hands such as
  pairs, two pair, straights, and flushes determine the base Chips and Mult.
  Scoring cards and other effects modify both values, producing a final score
  of `Chips × Mult`.
- **Clearing a Blind**: Scores from multiple hands accumulate within a Blind.
  Reaching the target Chips clears it; running out of hands ends the Run.
- **Discarding**: A discard replaces unwanted cards, using a limited resource
  to improve later hands.
- **Building**: Clearing a Blind awards money and opens the shop. Jokers change
  scoring, Planets level up poker hands, Tarot and Spectral cards modify the
  deck, Vouchers and Boosters provide longer-term upgrades, and rerolls spend
  money to refresh the shop.

The core challenge is allocating resources between clearing the current Blind
and growing the build over time: which hand to play, when to discard, which
Jokers to buy and how to order them, how much cash to keep, whether to skip a
Blind, and how to handle Boss rules. Together, these decisions form a complete
Balatro strategy.

## Bringing Balatro into EvoPolicyGym

In EvoPolicyGym v0.3.0, we integrated the unofficial Balatro engine
[Jackdaw](https://github.com/TylerFlar/jackdaw-balatro) as a vendored
dependency and used it to implement a Balatro evaluation Environment.

At every step, the Policy receives a public observation that includes:

- the Ante, Blind, target score, and remaining hands and discards;
- the hand, Jokers, Consumables, public deck statistics, and current poker-hand
  levels;
- the current shop, Boosters, Vouchers, Tags, and cash;
- a strict enumeration of legal Actions for the current phase;
- rule descriptions for visible objects in the pinned engine version.

The Policy returns a semantic Action such as playing, discarding, buying,
selling, rerolling, opening a pack, or reordering Jokers. The Environment does
not repair an invalid Action; it records a Policy failure. State may persist
within one Episode, while every new Episode starts with a fresh Policy
instance.

The final Policy score is:

```text
number of Blinds cleared + 1000 × whether the Run was completed
```

Each cleared Blind contributes one point, and completing the Run adds 1,000
points. This preserves completion as the final objective while giving Policies
that have not yet won continuous Feedback through their average progress.

## A brief overview of the EvoPolicyGym evaluation process

During an EvoPolicyGym Run, a Coding Agent starts from an initial Program,
repeatedly submits strategies, evaluates them on training Episodes, and
continues optimizing from scores and replays. After the Agent finishes,
Validation selects the final Program, and Assessment measures it on held-out
test Episodes. Every submitted Program, its Feedback and replays, and the
optimization record are retained in the Run data.

## Experiment

We compared the packaged baseline with the Programs handed off by optimization
Runs from three Coding Agents: Luna, Terra, and Sol. All three Agents started
from the same baseline and used the same training Episode pool and Environment
interaction budget.

The baseline is a deterministic poker-hand strategy. It enumerates every
one-to-five-card combination in the current hand and chooses according to
standard poker-hand rank and card values. In the shop, it buys the first
affordable Joker; when opening a pack, it selects the first Joker. It does not
use discards, Consumables, or rerolls, nor does it manage Joker combinations or
the economy.

Luna, Terra, and Sol each received the same training Episode pool and a total
Episode budget of 1,024. After optimization, we froze the Program selected by
each Agent and evaluated all four Programs with the `epg2` engine on the same
128 held-out test Episodes:

| Experiment | Agent | Reasoning | Training Episode budget | Test Episodes |
| --- | --- | --- | ---: | ---: |
| Packaged baseline | — | — | 0 | 128 |
| Luna | `gpt-5.6-luna` | `xhigh` | 1024 | 128 |
| Terra | `gpt-5.6-terra` | `xhigh` | 1024 | 128 |
| Sol | `gpt-5.6-sol` | `xhigh` | 1024 | 128 |

The experiment used Red Deck, White Stake, a Run seed of `20260729`, and a
60-second timeout for each Episode.

## Results

<figure class="blog-result-figure">
  <img
    src="../../images/blog/balatro-heldout-results-en.svg"
    alt="Balatro held-out test results for four Programs. Sol cleared 10.45 Blinds on average, completed 5 of 128 Runs, and achieved a mean final score of 49.52."
    loading="lazy"
    decoding="async"
  />
</figure>

Sol was the only Policy to complete a Run. It cleared 10.45 Blinds on average,
2.83 times the baseline result, and completed 5 of 128 Runs. Luna and Terra
also more than doubled the baseline's average progress, but neither completed
a Run. Reducing early mistakes can extend a Run; completing one requires a
coherent long-term build strategy.

## The baseline's capability boundary

The packaged baseline enumerates every one-to-five-card combination in the
current hand, preferring combinations with a higher standard poker-hand rank
and card values. In the shop and when opening packs, it selects the first Joker
that meets its conditions.

It answers “which hand ranks higher right now,” but does not connect projected
score, discards, Joker combinations, economy, and Boss rules.

## What strategy did Sol build?

Sol built an approximate model of the game state and applied a coordinated set
of heuristics for each phase. The final Policy changed in three main ways.

### Score estimation and hand planning

Sol still enumerates one-to-five-card combinations, but evaluates their
expected `Chips × Mult`. The estimate accounts for poker-hand levels, scoring
cards, Enhancements, Editions, held-in-hand effects, and Joker combinations.

The Policy combines that estimate with the target score and the remaining
hands and discards to decide whether to play immediately or keep searching. It
protects cards with value while held in hand and, when no discards remain, can
play low-value cards to draw replacements.

### Joker construction and economy management

Sol estimates Joker value, seeks stable combinations of Chips, Mult, and X
Mult, and reorders Jokers according to effect dependencies. When all slots are
full, it replaces an existing component only if the candidate Joker is clearly
stronger.

Shop decisions are also tied to the build. Based on the Ante, available cash,
and missing components, the Policy decides whether to buy, preserve interest,
or reroll. Cash is used to strengthen later rounds, not just the current shop.

### Cross-round state management

The Policy records shop visits, pack openings, poker hands, and skipped Blinds,
then adapts its behavior to Boss rules. Score estimates influence discards,
discards affect the probability of clearing the Blind, and the resulting
economy changes the next round's build, forming a complete decision loop.

## How models use Environment Feedback

Given the same baseline, training data, and budget, Sol was the only model to
complete a Run. It used replay and score Feedback more effectively, organizing
local lessons into a complete strategy spanning hands, construction, and
economy—more like an experienced player.

Sol's Policy also reveals an engineering problem: about 1,860 lines of logic
are concentrated in a single file, creating tight coupling. The next step is
to separate the strategy into modules that are easier to test, calibrate, and
continue improving.

## Next steps

The first direction is Skills. An effective Skill can provide more than domain
knowledge: it can also provide methods for modularization, replay analysis, and
testing. Comparing the same Agent and budget with and without a Skill would
measure both the final score and the engineering structure of the Policy.

The second direction is cross-Environment RL. The goal is not to memorize one
Environment's rules, but to learn how to locate failures, form hypotheses,
design experiments, and update strategies—and transfer that process to an
unseen Environment.

While integrating different Environments, we also encountered another
valuable but underexplored question: can an Agent observe an external system
and build an Environment suitable for training and evaluation? For example,
can an Agent understand a game's core rules and implement a behaviorally
equivalent engine that strips away strategy-irrelevant details such as art and
audio, retaining only states, Actions, and Feedback?

Answering this requires measuring whether an Agent can correctly abstract
states, Actions, Feedback, and evaluation rules. Today we primarily evaluate
how Agents use Environments to optimize strategies; Environment construction
itself still lacks systematic measurement. If an Agent can do this well, its
Environment can be integrated into EvoPolicyGym and used for further strategy
optimization. This complete loop—from observation to modeling to
optimization—could help future Agents adapt to new Environments faster and
build effective decision systems.

## Code and notes

- [EvoPolicyGym Balatro Benchmark](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/jackdaw/balatro)
- [EvoPolicyGym](https://github.com/Linzwcs/EvoPolicyGym)

This Benchmark is unaffiliated with LocalThunk, Playstack, or the official
Balatro project and contains no official card faces, artwork, music, fonts, or
other game assets.
