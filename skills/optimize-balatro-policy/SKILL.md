---
name: optimize-balatro-policy
description: Build, modularize, improve, test, and select a reward-aligned EvoPolicyGym Bot system for the Jackdaw Balatro Benchmark. Use when architecting or refactoring program/, assigning module ownership and dependency direction, choosing high-value strategy experiments, diagnosing or optimizing layered state/mechanics/outcome/value/policy models, analyzing indexed train Feedback or replay.jsonl, modeling hands, draws, builds, economy, and visible effects, hardening legal Actions, partitioning a train-only Episode pool and budget, running matched Program comparisons, diagnosing Policy failures, or handing frozen published candidates to Host Validation and Assessment.
---

# Optimize Balatro Policy

Build a coherent Policy system that can win complete runs. Treat Balatro
strategy, software architecture, replay testing, and noisy evaluation as one
engineering problem. Prefer a small shared decision model over disconnected
phase heuristics, tooltip tier lists, and encounter-specific patches.

## Load the reusable resources

- Before planning or interpreting an evaluation, read
  [references/experiment-protocol.md](references/experiment-protocol.md).
- Before choosing the next capability, use the failure-signature routing and
  experiment cards in
  [references/experiment-catalog.md](references/experiment-catalog.md).
- Before editing `program/`, read the applicable implementation pattern in
  [references/implementation-playbook.md](references/implementation-playbook.md).
- Before extracting, splitting, merging, or rewiring Policy modules, read
  [references/modularity-guide.md](references/modularity-guide.md).
- Before changing a prediction, value, planning, or selection model, localize
  the lowest incorrect layer with
  [references/modeling-stack.md](references/modeling-stack.md).
- Before changing scoring, discard, shop, build, or Joker-order behavior, read
  the relevant part of
  [references/strategy-lessons.md](references/strategy-lessons.md).
- Use `scripts/summarize_evidence.py` to pool public submission Feedback by
  immutable digest. Do not hand-calculate repeated evidence summaries.
- Use `scripts/compare_indexed_feedback.py` to compare separate control and
  candidate Feedback files. It refuses unmatched Episode-index sets or repeat
  counts.

## Align with the objective

- Distinguish Benchmark reward from in-game Chips and dollars. Optimize
  expected Episode score: Blinds cleared plus the large complete-run win bonus.
- Keep win reward and ordinary progress separate. A normal win clears 24
  Blinds and receives the 1000-point win bonus, producing run reward 1024.
- Treat Ante reach and Blinds cleared as development diagnostics, not
  substitutes for winning. Prefer changes that increase robust Ante 8
  completion probability over changes that only improve early progress.
- Group evidence by Program digest and pool all Episodes for that digest. A
  single win is high-variance evidence, not proof that one observed build or
  purchase caused it.
- Record wins, Episodes, Policy failures, mean/median Blinds, early deaths,
  mid/late tails, and Ante reach separately. Never compare only the largest
  submission mean. Results for the same selected Episode indices are matched;
  results for different index sets remain unmatched noisy evidence.

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

1. Read the Host task's Run-local Episode-index range, Episode budget,
   submission limits, and candidate limit. Before the first result, partition
   both train indices and spend into baseline, diagnosis, matched capability
   development, and frozen confirmation roles.
2. Inspect every file under `program/` and the permitted public train Feedback
   and advertised Artifacts. Record the unchanged Program digest and preserve
   an exact restore method.
3. Submit the smallest useful baseline batch to establish correctness,
   distribution, failures, Ante reach, and wins.
4. Keep an evidence ledger with digest, submission IDs, selected Episode
   indices, preassigned role, Episode count, mean/median Blinds, `≤5`, `≥12`,
   `≥18`, failures, wins, hypothesis, and decision. Keep the ledger in working
   reasoning; do not add non-Policy evidence files to the submitted `program/`.
5. Inspect replay trajectories and omission markers only for baseline,
   diagnosis, and capability-development submissions. Identify the first
   consequential bad decision in representative weak, strong, and failed
   Episodes.
6. Enter frozen confirmation only once. Stop editing strategic behavior,
   evaluate the frozen shortlist on pre-reserved, previously unseen train
   indices, and inspect aggregate outcomes only.

All Agent-visible submissions are train evidence from one fixed Host-owned
Episode pool. The Agent can choose and reuse Run-local Episode indices but
cannot observe their actual Environment or Policy seeds. Reusing an index
preserves its hidden Episode specification and Policy seed while creating a
fresh Environment and Policy runtime and consuming budget again. Use identical
index selectors for matched online A/B, and treat comparisons over different
selectors as unmatched. Follow the complete index-partition, budget, and freeze
procedure in the experiment protocol.

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

Treat this tree as a responsibility map, not a mandatory file list. Adapt names
and merge adjacent small modules according to the ownership and dependency
rules in the modularity guide, but preserve these enforceable boundaries:

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
- Give every responsibility one authoritative owner and keep dependencies
  acyclic toward normalized facts and pure models. Do not preserve parallel
  scorers, parsers, tier lists, or Action builders after migration.

Use this decision pipeline:

```text
observation -> StateView + LegalCatalog
            -> MechanicsSnapshot + EpisodePlan
            -> OutcomeEstimate -> ActionValue
            -> ranked Intent -> ActionGateway -> Action
```

Treat the system checkpoint as required work, even though refactoring alone
does not increase reward. Keep modular refactors separate from strategic
experiments. Validate intent and Action equivalence on public replays, then use
a small matched submission only when local evidence cannot cover runtime
behavior. Do not defer structural work until the final evidence phase.

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
data cannot express an encountered rule. Diagnose the earliest wrong model
layer and fix it there; do not compensate for a mechanics or prediction defect
with an upper-layer threshold.

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
- Treat two hidden cards with absent rank or suit as two unknowns, never as a
  known Pair or Flush relation.
- Test the downstream selector whenever calibration changes. A more accurate
  immediate score can reduce survival when the play/discard horizon remains
  myopic.

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
- Separate scored-card, held-card, main additive-Mult, main XMult, retrigger,
  and copy phases. Do not apply a generic Joker sort across unresolved copy
  dependencies.

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

## Run controlled capability experiments

Change one capability at a time:

1. Route the observed failure signature through the experiment catalog. State
   which public states are eligible, which decisions may change, why the change
   should improve survival or growth, and what would reject it.
2. Run a development replay or counterfactual opportunity audit before an
   environment evaluation. Count eligible states and changed actions.
3. Predeclare a small development index selector. Evaluate the exact control
   digest and candidate digest on that same selector, then compare public
   outcomes by `episode_index`. Prefer narrow guards that prevent catastrophic
   early regressions.
4. When a candidate looks promising, repeat the matched comparison on a second
   predeclared selector or evaluate the unchanged digest on reserved train
   indices. Require evidence beyond one matched batch.
5. Require zero Policy failures and no safety regression. Restore rejected
   controls exactly and verify their digest.

Once frozen confirmation begins, do not inspect its replay to learn how to
patch. Use its aggregate evidence only to reject brittle candidates or order
the final shortlist. Host Validation and Assessment occur after Agent cleanup,
publish nothing back to the Agent workspace, and stay outside the iteration
loop.

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
`evopolicygym-session submit`; shell command sequencing must not continue
after the failure.

## Iterate by capability

Use a staged sequence unless evidence points elsewhere:

1. Establish the unchanged baseline and diagnose the first consequential error.
2. Eliminate Policy failures and implement exact Action admission.
3. Complete the required system checkpoint: normalized state, Episode plan,
   one-owner module boundaries, an acyclic dependency graph, and persistent
   replay tests.
4. Audit whether the proposed capability can alter actions often enough to
   matter.
5. Unify hand enumeration, visible scoring, draw probability, and discard
   value.
6. Add structured effect roles and build-aware scoring.
7. Add economy, slot replacement, pack, consumable, and ordering decisions.
8. Add Blind scaling, Boss constraints, and long-horizon build control.

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
- Reserve a meaningful confirmation allocation before development consumes the
  budget, including train indices that development will not inspect. Do not
  spend the whole reserve chasing one promising result.
- Reuse identical selectors deliberately for matched comparisons, remembering
  that every reuse consumes budget. Never inspect, infer, name, or claim
  control over the hidden seed identities behind Run-local indices.
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
  credible candidate set. The Host evaluates them on identical private
  Validation Episodes only after Agent exit; do not consume train budget
  pretending to reproduce that stage.
- Expect neither Validation nor held-out Assessment evidence in workspace
  Feedback. Never adapt after final handoff.
- Finish successfully before exiting. Unsubmitted workspace edits are not
  candidates.

Optimize for defeating the Ante 8 Boss. Use cleared Blinds and Ante reach as
development signals, but remember that the objective's large win bonus makes a
robust complete-run system more valuable than a brittle progress heuristic.
