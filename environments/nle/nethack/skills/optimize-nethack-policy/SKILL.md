---
name: optimize-nethack-policy
description: Improve an EvoPolicyGym Policy Program for the deterministic NLE NetHackScore Benchmark.
---

# Optimize an NLE NetHack Policy

Maximize mean shaped return in deterministic but hidden NetHackScore Episodes.
The task uses NLE 1.3.0, NetHack 3.6.7, a fixed neutral human male Monk, the
standard 23-action task set, and at most 5,000 Policy steps. Each Policy failure
receives a large negative return, so validity and termination come first.

## Respect the ABI

- Export `make_policy(context)` from `policy.py` and return an object with
  `act(observation)`.
- Return an exact integer from 0 through 22. Booleans, floats, containers, and
  out-of-range integers are invalid; the Benchmark never repairs an Action.
- A fresh Policy process and instance are created for every Episode. Keep map,
  inventory, prompt, and plan state only between `act()` calls in that Episode.
- Read public static configuration from `context.environment_parameters`.
  Never infer, request, or search for Environment seeds or Host paths.

## Actions

```text
 0 more               8 northwest         16 run_northwest
 1 north              9 run_north         17 up
 2 east              10 run_east          18 down
 3 south             11 run_south         19 wait
 4 west              12 run_west          20 kick
 5 northeast         13 run_northeast     21 eat
 6 southeast         14 run_southeast     22 search
 7 southwest         15 run_southwest
```

Actions are NLE keyboard inputs. In a prompt, the same raw key can mean an
answer rather than movement. For example Actions 1 through 8 emit `k l j h u n
b y`; Action 21 emits `e`, Action 22 emits `s`, and Action 0 emits Enter. Inspect
`message` and `input_mode` before interpreting an Action as an ordinary command.

## Observation

The observation is a dictionary:

- `screen.glyphs`: `int16` `TensorValue`, shape `(21, 79)`;
- `screen.chars`: `uint8` `TensorValue`, shape `(21, 79)`, row-major visible
  character codes;
- `screen.colors`: `uint8` `TensorValue`, shape `(21, 79)`;
- `stats`: named public NLE blstats, including `x`, `y`, `score`, HP, depth,
  gold, experience, turn, hunger, encumbrance, dungeon level, and conditions;
- `message`: the current public NetHack message;
- `inventory`: only populated entries with `letter`, `description`, `glyph`,
  and `object_class`;
- `input_mode`: `normal`, `yes_no`, `get_line`, or `more`.

The Policy does not receive NLE's privileged `internal` array, raw seeds,
ttyrec paths, rewards, or evaluator state. Decode tensors from
`TensorValue.data`; do not import or reach into the trusted NLE Environment.

## Build an Episode-local controller

Use explicit layers that can be tested separately:

1. **Prompt handler:** respond to `<More>`, direction, eating, and yes/no
   questions before normal navigation. A prompt that is mistaken for movement
   commonly creates frozen-step penalties or loops.
2. **Map memory:** combine visible characters, colors, glyphs, current position,
   and dungeon level. Track visited, blocked, dangerous, door, corridor,
   staircase, item, and monster cells without assuming a hidden seed route.
3. **Immediate safety:** monitor HP, hunger, conditions, nearby monsters, and
   escape space. Do not start a long run or repeated search while threatened.
4. **Exploration:** prefer reachable low-visit frontiers, search near plausible
   dead ends, handle doors deliberately, and avoid immediate reversal loops.
5. **Resource state:** track public inventory descriptions and observed outcome
   messages. Initiating `eat` and selecting an inventory letter are separate
   decisions.
6. **Progress plan:** descend when prepared, collect useful public items, and
   balance score gain against survival and frozen-step penalties.

The standard 23-action profile is deliberately narrower than a full NetHack
keyboard. Do not write plans that require unavailable commands such as wield,
wear, quaff, read, open, or arbitrary inventory letters.

## Optimize the actual metric

The scalar score is mean Episode return from `NetHackScore-v0`: NetHack score
deltas plus a `-0.01` penalty when repeated Environment steps do not advance
game time. `mean_game_score`, depth, deaths, ascensions, truncations, and Policy
failures are secondary diagnostics. A Policy failure receives
`-max_episode_steps`, not the partial return accumulated before failure.
`frozen_steps`, `mean_frozen_steps`, `frozen_step_fraction`, and
`mean_frozen_penalty` expose the exact aggregate contribution of unchanged-turn
Actions to this score; they do not select or summarize trajectory content.

Small batches have high procedural variance. First eliminate failures and
frozen prompt loops, then compare unchanged candidates on equal larger batches.
Use 1--4 Episodes per search submission until memory and runtime are measured.

## Use Feedback safely

Training Feedback includes aggregate scores plus complete raw evidence for all
Episodes in that submission. Read `artifact-manifest.json`, then use its
ordinals to align every `trajectory-*.jsonl.gz` transition with the exact
Policy-visible observations in `observations-*.npz`. Open NPZ files with
`numpy.load(..., allow_pickle=False)`. No Environment or Host code has already
selected frames, created screenshots, rendered videos, or decided which
segments are important.

Choose your own analysis. You may decode terminal `chars` and `colors`, inspect
glyphs or named stats, render selected observations with Pillow, export a
segment with ImageIO, compare messages and inventory, or use another derived
representation. Keep non-submitted scripts, selected frames, derived images,
and notes under `analysis/`; only `program/` is submitted. Bulk evidence from
older submissions can be evicted, while the newest complete submission is
protected, so copy only the derived material you decide is useful before a
later submission.

Validation and Assessment are Host-only aggregate phases and publish no
detailed Artifacts back to the workspace. Never attempt to recover hidden
seeds, Host paths, ttyrec data, or privileged simulator state from Feedback.
Treat small-batch score movement as noisy and use equal Episode counts when
comparing candidates.

Before `finish`, compare the candidate Program digests and the actual source
changes you intended. Submit behaviorally distinct candidates that test real
strategy alternatives. Renaming variables, retaining unused branches, or
passing a flag that does not change reachable behavior wastes a Validation
slot even though it creates a different source digest. Exact aggregate ties on
the same hidden Validation Episodes are a reason to inspect candidate
distinctness, not evidence that the Host should inspect or rewrite Programs.

## Iterate in stages

1. Submit the packaged baseline on a small batch and inspect the manifest.
2. Select and decode the observations or trajectory segments needed to explain
   failures and repeated behavior.
3. Make prompt handling total and eliminate invalid Actions/timeouts.
4. Add persistent dungeon-level map memory and loop detection.
5. Improve frontier choice, door/search behavior, and staircase handling.
6. Add HP, hunger, inventory, monster, and condition-aware interrupts.
7. Remove behaviorally redundant candidates, then compare promising distinct
   candidates with equal Validation budgets before final handoff.
