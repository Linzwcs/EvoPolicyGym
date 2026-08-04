---
locale: en
page: nethack-policy-evolution
title: "Into the Dungeon: Building Exploration Systems for NetHack"
description: "How coding agents turned complete NetHack trajectories into executable Policies for navigation, obstacle handling, and dungeon progress."
lead: "Coding agents used Policy-visible NetHack trajectories to build spatial memory, recover from failed movement, and make measurable progress into the dungeon."
publishedAt: "2026-08-03"
date: "2026-08-03"
authors: [evopolicygym]
tags:
  - Benchmark
  - NetHack
  - Experiment
  - Policy Evolution
status: published
---

## What is NetHack?

NetHack is a turn-based roguelike set in a procedurally generated dungeon. The
full game asks the player to descend, obtain the Amulet of Yendor, return to the
surface, and complete an ascension. Reaching that goal requires much more than
winning individual fights: the player must explore unknown layouts, interpret
messages, manage resources, remember useful locations, and survive permanent
death.

<!-- truncate -->

A simplified progression loop looks like this:

```text
explore a dungeon level
        ↓
handle creatures, obstacles, and resources
        ↓
find a downward staircase
        ↓
descend and repeat
```

The details change in every Episode. Rooms and corridors are rearranged,
objects and creatures appear in different places, and only part of the current
level is visible. An Action that looks reasonable locally may waste hundreds
of turns, consume scarce food, or lead the character away from the route it
was trying to follow.

This makes NetHack a useful Environment for studying executable strategy. A
Policy must combine immediate reactions with memory and longer-term goals, and
it must recognize when the world did not respond as expected.

## Bringing NetHack into EvoPolicyGym

The Benchmark integrates NLE 1.3.0 `NetHackScore-v0`, backed by NetHack 3.6.7.
The Policy receives a semantic view of the terminal map together with status
values, the current message, public inventory entries, and the current input
mode. It chooses from 23 Actions covering movement, running, stairs, waiting,
kicking, eating, searching, and message prompts.

The Action profile is intentionally narrower than the complete NetHack command
set. Within a 5,000-step Episode, it concentrates the experiment on early-game
exploration, obstacle handling, survival, and descent rather than full-game
ascension.

The score follows progress recognized by NetHack and penalizes repeated steps
that leave the character frozen in place. This gives the Agent a primary
optimization signal while dungeon depth, game score, and frozen-step rate help
explain what kind of behavior produced it.

## Learning from complete trajectories

Many weak NetHack Policies do not crash. They continue running while pushing
against a boulder, trying to cross iron bars, alternating between two tiles, or
standing on a staircase without descending. An aggregate score says that the
Policy performed poorly, but not where its behavior broke down.

For each training submission, the Environment therefore returns the complete
Policy-visible trajectory for every evaluated Episode. The Agent can inspect
positions, Actions, messages, status changes, and repeated states, then connect
those patterns back to the source code.

```text
submit an executable Policy
        ↓
inspect complete training trajectories
        ↓
identify a behavioral failure
        ↓
rewrite memory, routing, or interaction rules
        ↓
submit a new Program
```

The Environment supplies evidence, not a diagnosis. Deciding which Episodes
to inspect and which patterns matter remains part of the coding agent's work.
Lasting improvement is stored in the executable Program rather than in model
weights or hidden state carried between Episodes.

## Experiment

We ran three GPT-5.6 model variants—Luna, Terra, and Sol—through Codex. Each
Agent began with the same packaged baseline and could improve it using training
scores and trajectories. The optional NetHack optimization Skill was disabled,
so the Agents had to develop their own analysis and revision process.

The baseline already performs simple local exploration. It remembers visit
counts, usually prefers a less-visited neighboring tile, avoids immediate
reversal when alternatives exist, searches periodically, kicks visible closed
doors, and eats recognized food when hungry. It does not build an explicit map,
plan routes to distant targets, remember failed edges, or treat descending as a
persistent objective.

| Setting | Value |
| --- | --- |
| Environment | NLE 1.3.0 · NetHack 3.6.7 |
| Task | `NetHackScore-v0` · 23 Actions |
| Episode limit | 5,000 Policy steps |
| Training allowance | up to 128 Episodes |
| Final Assessment | 256 held-out Episodes |
| Optional NetHack Skill | disabled |

The training allowance was a ceiling rather than forced consumption. Sol used
all 128 Episodes, Terra used 68, and Luna used 40 before finishing.

## Results

Sol produced the strongest selected Policy in this experiment. It reached a
mean Assessment return of `204.026`, a mean game score of `208.230`, and a mean
dungeon depth of `2.867`. Its deepest held-out Episode reached depth 11.

| Agent lane | Training used | Submissions | Assessment return | Mean game score | Mean / max depth | Frozen steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GPT-5.6 Sol + Codex** | 128 / 128 | 8 | **204.026** | **208.230** | **2.867 / 11** | **27.53%** |
| GPT-5.6 Terra + Codex | 68 / 128 | 4 | 80.237 | 87.094 | 1.082 / 4 | 33.77% |
| GPT-5.6 Luna + Codex | 40 / 128 | 4 | 63.773 | 70.777 | 1.094 / 4 | 35.75% |

All three selected Policies completed the held-out Assessment without a Policy
execution failure. None ascended. The results measure early-game exploration,
survival, and dungeon progress—not complete NetHack mastery.

![Complete semantic replay of a NetHack training Episode from Sol's selected
Policy. The replay covers all 1,269 Policy steps, reaches dungeon depth 11 and
a maximum game score of 860, and ends in death.](/images/blog/nle-sol-policy-training-replay.gif)

*Submission 000008, training Episode 16. This replay shows an Agent-written
Policy acting autonomously from the first observation to the end of the
Episode. It is a representative training trajectory, not part of the held-out
Assessment reported in the table.*

## The baseline's capability boundary

The baseline can answer a useful local question:

> Which visible neighboring tile has been visited least?

It cannot yet answer the larger navigation question:

> How should I build and maintain a route through an uncertain, multi-level
> dungeon?

That gap became the main opportunity for Policy evolution. Moving beyond the
baseline required a Policy that could preserve knowledge after a location left
the screen, pursue a target across several rooms, and revise its plan after a
failed Action.

## What strategy did Sol build?

Sol turned the local exploration baseline into a more structured navigation
system. Its final Policy combined three ideas that reinforce one another.

### Persistent spatial memory

The Policy records discovered terrain, routes, targets, and movement outcomes.
Instead of repeatedly choosing only among adjacent tiles, it can use earlier
observations to navigate through known rooms and corridors. Information remains
useful after the character moves elsewhere on the level.

### Goal-directed exploration

Sol made dungeon progress an explicit objective. When the Policy knows about a
downward staircase, it can preserve that target, route back to it, and use it.
When it does not know a staircase, it seeks unexplored space rather than merely
choosing the least-visited visible neighbor.

This turns depth from a metric observed after the Episode into a goal expressed
inside the executable strategy.

### Failure detection and recovery

Movement in NetHack can fail because of walls, boulders, iron bars, creatures,
doors, or a stale interpretation of the map. Sol compares the intended move
with the next observation. If the expected transition did not occur, the Policy
can mark the route as blocked, discard an invalid target, and choose another
path instead of repeating the same Action.

The result is more general than a list of special cases: observe the outcome,
update the internal map, and replan.

## How agents used Environment Feedback

**Luna — remembering obstacles.** Luna found trajectories dominated by repeated
attempts to move through a boulder or iron bars. It added memory for failed
directions and reduced immediate retries against static obstacles.

**Terra — escaping loops and using stairs.** Terra introduced anti-loop
behavior and explicit stair descent. Its strongest selected candidate came
from an earlier revision, illustrating that continued editing does not always
produce a better Policy.

**Sol — organizing fixes into a navigation system.** Sol combined remembered
routes, stair-oriented progress, blocked-edge handling, and target recovery.
Rather than treating each failure as an isolated patch, it connected the
lessons through a shared model of position, routes, targets, and outcomes.

In every case, the durable result was not the Agent's explanation of the
trajectory. The lesson had to survive as code that independently received
observations and returned Actions during held-out evaluation.

## Findings and boundaries

Complete semantic trajectories were enough for the Agents to locate behavioral
failures and make useful Policy changes without an Environment-authored
diagnosis. The experiment also shows why NetHack needs more than one metric:
return ranks candidates, while depth, game score, and frozen-step rate reveal
different aspects of exploration and progress.

These results are still an initial study. The Agents did not consume equal
training budgets, and each model lane is represented by one primary Run. With
the short 128-Episode training ceiling, coding-agent search can vary because of
randomness: a separate Terra Run under the same Environment configuration
scored `49.464`, compared with `80.237` here.

The table should therefore be read as evidence that the Agents built better
NetHack Policies in these Runs, not as a general ranking of Luna, Terra, and
Sol.

## Code and notes

- [NLE NetHack Benchmark](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/nle/nethack)
- [Evaluation and Runs](/docs/evaluation/)
- [Policy boundary](/docs/policy/)

The EvoPolicyGym adapter is MIT licensed. NLE and NetHack remain separate
dependencies governed by their respective licenses.
