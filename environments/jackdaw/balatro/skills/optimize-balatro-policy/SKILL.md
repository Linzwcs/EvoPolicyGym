---
name: optimize-balatro-policy
description: Design, improve, test, and select robust EvoPolicyGym Policy Programs for the Jackdaw Balatro Benchmark. Use when architecting any file under program/, analyzing Feedback or replay.jsonl, modeling hands and visible card effects, hardening legal Actions, allocating the Episode budget, diagnosing Policy failures, or handing final Balatro candidates to the Host.
---

# Optimize Balatro Policy

Build a coherent Policy system that can win complete runs. Treat Balatro
strategy, software architecture, replay testing, and noisy evaluation as one
engineering problem. Prefer a small shared decision model over disconnected
phase heuristics, tooltip tier lists, and encounter-specific patches.

## Preserve the boundary

- Edit only `program/`; treat `feedback/` as read-only evidence.
- Use only the Benchmark specification, public Feedback and Artifacts, the
  current observation, and `legal_actions`.
- Do not inspect or import Jackdaw, Benchmark, Host, seed, or hidden Case
  internals even when local process execution makes them reachable.
- Never predict a hidden seed, future draw, shop roll, or private Case identity.
- Re-read every entity and hand index from the current observation. Indices are
  ephemeral.
- Keep learning outside Episodes. Retain only Episode-local intent or plan
  state in the Policy instance; never update persistent parameters in `act()`.

## Establish evidence first

1. Read this skill, inspect every file under `program/`, and inspect all
   existing Feedback and advertised Artifacts.
2. Submit the unchanged Program with the smallest useful batch to establish its
   correctness, score distribution, failure count, Ante reach, and win rate.
3. Keep an evidence ledger with submission ID, digest, Episode count, score,
   failures, wins, Ante reach, hypothesis, and architectural change.
4. Inspect retained replay trajectories and omission markers, not only
   terminal states or aggregate scores. Identify the first consequential bad
   decision in representative weak, strong, and failed Episodes.

Treat small-batch score changes as noisy observations. Compare repeated
identical digests and distributions before attributing a result to an edit.

## Build a Policy system

Keep `policy.py:make_policy` as the ABI entrypoint, but use the whole
`program/` directory. Once behavior extends beyond the baseline, separate
responsibilities into a structure equivalent to:

```text
policy.py          ABI adapter and composition
policy_system/
  state.py         observation normalization and derived state
  actions.py       legal-action catalog and exact Action construction
  hands.py         hand enumeration, scoring, and draw value
  effects.py       visible card-effect interpretation
  planning.py      build, economy, and long-horizon planning
  strategy.py      phase planners and final decision
```

Adapt names and merge genuinely small modules, but preserve these boundaries:

- Concentrate raw dictionary access and nullable-field normalization in one
  observation adapter.
- Make hand evaluation and card-effect evaluation pure wherever practical.
- Make every phase planner return an intent or ranked candidate, not an
  unchecked Action object.
- Route every emitted Action through one legal-action gateway.
- Use one shared value model for playing hands, discarding, buying, selling,
  opening packs, using consumables, and planning the build.
- Represent unknown effects explicitly and discount confidence instead of
  silently treating them as zero or inventing behavior.

Use this decision pipeline:

```text
observation -> normalized state -> Episode plan -> phase candidates
            -> shared value model -> legal-action gateway -> Action
```

Do not preserve a monolith merely to make small diffs. Refactor an abstraction
when several decisions need the same rule, while keeping each submission tied
to one testable capability.

## Enforce the Action contract

- Index `legal_actions` by `kind` on every call.
- Construct exactly the fields described for the selected kind; never emit
  descriptor metadata as Action fields.
- Validate target indices against the current descriptor, not only against the
  corresponding observation list.
- Read nested `targets` for `use_consumable` and `pick_pack_card`, including
  each target's allowed `card_indices`, minimum, and maximum.
- Preserve selected-card order when `selection_order_matters` is true.
- If the preferred intent is unavailable, rank a different currently legal
  intent. Never guess, repair, or reuse a stale index.
- Treat any `invalid_action`, exception, timeout, or protocol failure as a
  release-blocking correctness defect, not a bad score.

## Model decisions consistently

### Hands and draws

- Enumerate legal one-to-five-card plays and classify their actual hand types.
- Start from visible `poker_hands` Chips and Mult, then account for card Chips,
  enhancements, editions, seals, debuffs, played-card order, and modeled Joker
  effects.
- Carry an approximation confidence when an effect cannot be modeled exactly.
- Compare playing now with discarding by expected improvement, remaining
  hands, remaining discards, visible rank and suit counts, outs, and Blind
  target pressure.
- Let the established build alter hand value. Do not rank a level-one Straight
  above a highly upgraded core hand merely by poker category.

### Effects and build

- Prefer structured `rule.parameters` and mutable `ability` values over text.
  Use `rule.summary` as semantic evidence or a localized fallback, not as a
  global substring-based tier list.
- Classify visible effects into roles such as base Chips, additive Mult,
  multiplicative Mult, retrigger, scaling, economy, hand-specific synergy,
  deck shaping, and defense.
- Track a primary and optional secondary hand plan, covered scoring roles,
  scaling trajectory, activation reliability, and switching cost.
- Make purchase ranking agree with the scorer and planner. Do not buy an effect
  the Policy cannot activate or value.
- Model temporary, scaling, conditional, consumable, and decaying effects
  differently. Evaluate Joker and played-card ordering when resolution order
  matters.

### Economy and horizon

- Price purchases by marginal contribution to the current build, activation
  probability, opportunity cost, interest lost, slot pressure, and survival
  need.
- Preserve an economy after acquiring enough early scoring to survive. Spend
  aggressively only when current or upcoming Blind pressure justifies it.
- Compare growth against upcoming Blind scaling rather than optimizing only
  the next Blind.
- Inspect active Boss rules and visible skip rewards explicitly. Treat Boss
  handling as a constraint on the plan, not a late special case.

## Add replay regression gates

Before every submission:

1. Run `python -m compileall -q program`.
2. Run any Policy-local unit tests.
3. Replay all available public observations with non-empty `legal_actions`
   through the pure decision core, grouped sequentially by Episode when testing
   Episode-local memory.
4. Assert that decisions do not raise, remain reproducible for identical
   context, state, and memory, and compile to an Action admitted by the current
   descriptor.
5. Retain every newly failing public state as a small regression fixture or
   invariant test.

Test shop, pack, consumable, full-slot, zero-money, no-discard, face-down,
debuffed, ordering, and Boss paths. Do not submit a known failing path merely
because its mean score is promising.

## Iterate by capability

Use a staged sequence unless evidence points elsewhere:

1. Eliminate Policy failures and centralize legal Action construction.
2. Establish the normalized state and replay regression harness.
3. Unify hand enumeration, immediate scoring, and discard value.
4. Add structured effect roles and build-aware scoring.
5. Add economy, slot replacement, pack, consumable, and ordering decisions.
6. Add Blind scaling, Boss constraints, and long-horizon build control.

Change one decision capability per experiment. Avoid named-object patches when
the failure reveals a missing role, invariant, or abstraction. Revert
underperforming experiments exactly instead of approximately reconstructing a
previous candidate.

## Allocate evidence and finish

- Use small batches for correctness and directional experiments, then larger
  batches or repeated identical digests for candidate confidence.
- Keep more than one candidate only when they represent credible, robust
  alternatives rather than lucky observations.
- Follow the Host task's current `finish` syntax and candidate limit. When
  private Validation is configured, hand the Host the strongest ordered
  candidate set; do not consume Agent search budget pretending to reproduce
  Host Validation.
- Expect neither Validation nor held-out Assessment evidence in workspace
  Feedback. Never adapt after final handoff.
- Finish successfully before exiting. Unsubmitted workspace edits are not
  candidates.

Optimize for defeating the Ante 8 Boss. Use cleared Blinds and Ante reach as
development signals, but remember that the objective's large win bonus makes a
robust complete-run system more valuable than a brittle progress heuristic.
