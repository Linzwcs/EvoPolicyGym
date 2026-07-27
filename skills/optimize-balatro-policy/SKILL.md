---
name: optimize-balatro-policy
description: Build, improve, test, and select a reward-aligned EvoPolicyGym Bot system for the Jackdaw Balatro Benchmark. Use when architecting files under program/, analyzing Feedback or replay.jsonl, modeling hands, draws, builds, economy, and visible effects, hardening legal Actions, allocating the Episode budget, diagnosing Policy failures, or handing final candidates to the Host.
---

# Optimize Balatro Policy

Build a coherent Policy system that can win complete runs. Treat Balatro
strategy, software architecture, replay testing, and noisy evaluation as one
engineering problem. Prefer a small shared decision model over disconnected
phase heuristics, tooltip tier lists, and encounter-specific patches.

## Align with the objective

- Distinguish Benchmark reward from in-game Chips and dollars. Optimize
  expected Episode score: Blinds cleared plus the large complete-run win bonus.
- Treat Ante reach and Blinds cleared as development diagnostics, not
  substitutes for winning. Prefer changes that increase robust Ante 8
  completion probability over changes that only improve early progress.
- Group evidence by Program digest and pool all Episodes for that digest. A
  single win is high-variance evidence, not proof that one observed build or
  purchase caused it.
- Record wins, Episodes, Policy failures, score distribution, and Ante reach
  separately. Never compare only the largest submission mean.

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

After the unchanged baseline and first minimal tactical correction, establish
the Bot-system boundaries before adding a second strategic capability. Do not
keep extending a monolithic `policy.py`. Keep `policy.py:make_policy` as the
small ABI and composition entrypoint, and use the whole `program/` directory:

```text
policy.py          ABI adapter and composition only
policy_system/
  state.py         normalized StateView and EpisodePlan
  actions.py       legal catalog, Action construction, exact admission
  hands.py         hand classification and candidate enumeration
  scoring.py       visible scoring model and approximation confidence
  draws.py         outs, draw probability, and discard value
  effects.py       structured visible-effect roles
  planning.py      build, economy, Boss, and horizon model
  strategy.py      phase candidates and final decision
tests/
  test_replays.py  persistent public replay regression
```

Adapt names and merge adjacent small modules, but preserve these enforceable
boundaries:

- Concentrate raw dictionary access and nullable-field normalization in one
  observation adapter.
- Keep an explicit Episode plan containing the current primary and secondary
  hand plans, effect roles, economy phase, scaling trajectory, Boss constraints,
  and confidence. Update only Episode-local state.
- Make hand evaluation and card-effect evaluation pure wherever practical.
- Make every phase planner return an intent or ranked candidate, not an
  unchecked Action object.
- Construct and admit every emitted Action through one legal-action gateway.
- Use one shared value model for playing hands, discarding, buying, selling,
  opening packs, using consumables, and planning the build.
- Represent unknown effects explicitly and discount confidence instead of
  silently treating them as zero or inventing behavior.
- Persist the replay harness and regression fixtures under `program/`; do not
  leave essential tests only in one-off shell snippets.

Use this decision pipeline:

```text
observation -> StateView -> EpisodePlan -> phase intents
            -> shared outcome/value model -> ActionGateway -> Action
```

Treat the system checkpoint as required work, even though refactoring alone
does not increase reward. Validate behavioral equivalence on public replays,
then use a small submission to confirm it. Do not defer structural work until
the final evidence phase.

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

Build a reusable model of visible game mechanics, not a collection of
encounter-name patches. Parse structured rule parameters into effect roles and
evaluate actions as changes to survival probability, build strength, economy,
and win probability. Use localized text handling only when structured public
data cannot express an encountered rule.

### Hands and draws

- Enumerate legal one-to-five-card plays and classify their actual hand types.
- Start from visible `poker_hands` Chips and Mult, then account for card Chips,
  enhancements, editions, seals, debuffs, played-card order, and modeled Joker
  effects.
- Carry an approximation confidence when an effect cannot be modeled exactly.
- Estimate outs and draw probabilities from visible remaining-deck counts.
  Compare playing now with discarding by expected post-draw value, remaining
  hands and discards, current build activation, and Blind target pressure.
- Let the established build alter hand value. Do not rank a level-one Straight
  above a highly upgraded core hand merely by poker category.
- Compare predicted hand score with actual `last_hand` breakdowns and turn
  systematic error into effect-model regressions.

### Effects and build

- Prefer structured `rule.parameters` and mutable `ability` values over text.
  Use `rule.summary` as semantic evidence or a localized fallback, not as a
  global substring-based tier list.
- Classify visible effects into roles such as base Chips, additive Mult,
  multiplicative Mult, retrigger, scaling, economy, hand-specific synergy,
  deck shaping, and defense.
- Track a primary and optional secondary hand plan, covered scoring roles,
  scaling trajectory, activation reliability, and switching cost.
- Derive purchase, replacement, pack, consumable, and ordering values from the
  same effect and build model used by the hand scorer. Do not maintain a
  disconnected shop tier list.
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
- Evaluate a purchase or skip by its effect on the current plan and future
  survival, not by a fixed global threshold alone.

## Add replay regression gates

Before every submission:

1. Use fail-fast execution and run `python -m compileall -q program`.
2. Run any Policy-local unit tests.
3. Replay all available public observations with non-empty `legal_actions`
   through the pure decision core, grouped sequentially by Episode when testing
   Episode-local memory.
4. Pass every result through the same Action gateway used in production.
   Validate the exact field set, action kind, target membership, selected-card
   membership, uniqueness, order, and cardinality against the current
   descriptor. Checking only that the `kind` exists is insufficient.
5. Assert that decisions do not raise and remain reproducible for identical
   context, state, and Episode memory.
6. Compare modeled and observed hand outcomes where `last_hand` evidence is
   available.
7. Retain every newly failing public state as a small regression fixture or
   invariant test.

Test shop, pack, consumable, full-slot, zero-money, no-discard, face-down,
debuffed, ordering, and Boss paths. Do not submit a known failing path merely
because its mean score is promising. If any gate fails, stop before
`evopolicygym submit`; shell command sequencing must not continue after the
failure.

## Iterate by capability

Use a staged sequence unless evidence points elsewhere:

1. Establish the unchanged baseline and diagnose the first consequential error.
2. Eliminate Policy failures and implement exact Action admission.
3. Complete the required system checkpoint: normalized state, Episode plan,
   module boundaries, and persistent replay tests.
4. Unify hand enumeration, visible scoring, draw probability, and discard
   value.
5. Add structured effect roles and build-aware scoring.
6. Add economy, slot replacement, pack, consumable, and ordering decisions.
7. Add Blind scaling, Boss constraints, and long-horizon build control.

Change one decision capability per experiment, but implement it through the
shared system rather than appending a phase-local exception. Avoid named-object
patches when the failure reveals a missing role, invariant, or abstraction.
Revert underperforming experiments exactly instead of approximately
reconstructing a previous candidate.

## Allocate evidence and finish

- Treat the Episode budget as an optimization resource to consume, not merely
  a ceiling. Continue improving or measuring Programs until
  `episodes_remaining` reaches zero.
- Do not call `finish` while any Episode budget remains and another legal
  submission is possible. If no justified code change remains, spend the
  remainder re-evaluating the strongest digests to measure noise and improve
  candidate confidence.
- Size the final submission to consume the exact remaining budget without
  exceeding the per-submission limit. Stop early only when the Session prevents
  another legal submission, never because the current result merely looks good
  enough.
- Use small batches for correctness and directional experiments, then larger
  batches or repeated identical digests for candidate confidence.
- Enter repeat-only evidence collection only after the required system
  checkpoint is complete, replay tests persist, and no known correctness or
  model-consistency defect remains.
- Pool repeated evidence by digest. Estimate win frequency and ordinary
  progress separately; do not promote a candidate solely because one small
  batch contains a rare win.
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
