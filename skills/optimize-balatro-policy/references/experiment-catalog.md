# Experiment catalog

Choose experiments from observed failure signatures, not from a generic Joker
tier list. Run an opportunity audit before changing code, then use matched
train indices as defined in `experiment-protocol.md`.

## Contents

- Select the next experiment
- Correctness and observability
- Scoring and hand choice
- Draws and card throughput
- Build and shop planning
- Consumables and deck shaping
- Ordering and copy effects
- Bosses and late-game horizon
- Avoid low-value experiments

## Select the next experiment

Use terminal and replay evidence to identify the active bottleneck:

| Evidence signature | First experiment family | Likely model layer |
|---|---|---|
| Any Policy failure | Action admission and state transition | L0/L6 |
| High `≤5` count | early scoring or discard safety | L3–L5 |
| Median 10–11, weak `≥18` | build scaling, replacement, economy | L2/L4 |
| Frequent `≥18`, few wins | late horizon, Boss constraints, XMult | L1/L4/L5 |
| High cash at death | purchase threshold or bounded reroll | L4/L5 |
| Full weak Joker slots | marginal slot value and replacement | L2/L4 |
| Score prediction error | effect phases and calibration | L1/L3 |
| Many draws, weak final hands | outs and play/discard horizon | L3/L4 |
| Candidate rarely changes actions | opportunity audit or L5 integration | L5 |
| Gains plus catastrophic regressions | eligibility, confidence, fallback | L3–L5 |

Prefer the experiment with the largest combination of eligible states,
decision impact, expected late-run value, and testability. Do not implement
several rows at once. Confirm the suspected layer with
`modeling-stack.md` before writing code.

For every experiment predeclare:

- eligible public states;
- control and candidate digests;
- matched development selector;
- primary metric and safety metrics;
- expected action changes;
- rejection and exact-revert conditions.

## Correctness and observability

**Action contract hardening**

- Signal: `invalid_action`, exceptions, stale targets, or pack/consumable paths
  absent from tests.
- Change: centralize exact action construction and validate descriptor fields,
  target membership, uniqueness, cardinality, and selection order.
- Audit: replay every public state with nonempty legal actions.
- Gate: zero Policy failures. Do not trade correctness for score.

**State-transition reset**

- Signal: behavior depends on the preceding Blind, shop, or Episode.
- Change: reset round, shop, pending transaction, and consecutive-action state
  from public identity changes.
- Audit: replay two Episodes sequentially through one test harness while
  creating a fresh Policy instance per Episode.
- Reject: any state survives beyond its documented scope.

**Modular extraction checkpoint**

- Signal: duplicate raw parsing, Action construction, scoring, valuation, or a
  strategy orchestrator that owns unrelated mechanics and mutable state.
- Change: extract one pure leaf responsibility behind an explicit contract,
  route every caller to it, and remove the duplicate owner.
- Audit: compare the earliest moved seam, ranked intents, and final Actions on
  authorized public replay fixtures.
- Gate: require behavioral equivalence. Use matched same-index evidence only
  when replay cannot cover runtime composition, and expect score neutrality.
- Reject: any unexplained decision change, import cycle, second authoritative
  model, or bundled strategic threshold change.

## Scoring and hand choice

**Frequent effect calibration**

- Signal: systematic predicted/actual error in `last_hand`.
- Change: add the highest-frequency missing visible effect at the correct
  scored-card, held-card, main-additive, main-XMult, or retrigger phase.
- Audit: measure error before and after on development replay and count changed
  hand rankings.
- Guard: preserve unknown-effect confidence; avoid global text heuristics.
- Reject: calibration improves but matched Episode outcomes or safety worsen.

**Selector-horizon coupling**

- Signal: the numerically best immediate hand consumes the only useful draw or
  leaves an impossible Blind deficit.
- Change: rank actions by survival over remaining hands/discards, not one-hand
  score alone.
- Audit: compare immediate score, residual target, future draws, and hands
  remaining on changed states.
- Guard: retain a high-confidence lethal-hand fast path.

## Draws and card throughput

**Discard expected value**

- Signal: repeated low-value discards, chasing too few outs, or preserving the
  wrong made hand.
- Change: estimate visible outs and post-draw value using remaining-deck
  composition, build activation, and discard count.
- Guard: Boss legality, hidden-card uncertainty, and a survival threshold.
- Reject: early deaths rise even if average Blind improves.

**Safe kicker cycling**

- Signal: no discards remain, the current hand is nonlethal, and unused play
  slots could replace junk before a future hand.
- Change: append only low, unmodified cards outside the scoring hand, valuable
  draw structures, and held-card effects.
- Guard: activate only with future hands, adequate score margin, and no Boss
  conflict.
- Reject: any material Ante 1 regression; narrow eligibility before retesting.

## Build and shop planning

**Marginal Joker replacement**

- Signal: full slots, weak role coverage, cash available, and affordable shop
  inventory.
- Change: compare owned and offered Jokers in one shared build-value model;
  execute replacement as a sell intent followed by a revalidated purchase.
- Audit: count eligible shops, planned sales, completed purchases, and cleared
  stale intents.
- Guard: demand a margin covering sell loss, interest, uncertainty, and slot
  disruption.

**Bounded paid reroll**

- Signal: full slots, large cash surplus, no qualifying purchase, weak
  projected scaling.
- Change: permit one paid search action per shop above a reserve and Ante
  threshold.
- Guard: explicit per-shop counter and survival reserve.
- Reject: early economy or median progress worsens; do not add repeated rerolls
  before the one-reroll experiment passes.

**Role-completion valuation**

- Signal: builds contain redundant Chips or additive Mult but lack reliable
  multiplicative growth.
- Change: value marginal role coverage, activation, scaling rate, and switching
  cost rather than names.
- Guard: do not discard a mature engine for a speculative rare effect.

## Consumables and deck shaping

**Established-hand Planet use**

- Signal: a stable primary hand exists and a matching Planet is legal and
  affordable.
- Change: value permanent level growth against reserve and build maturity.
- Guard: require explicit public hand match; do not guess targets.

**Targeted Tarot or Spectral use**

- Signal: legal target descriptors and visible effect semantics identify a
  beneficial transformation.
- Change: generate targets through the Action gateway and score the resulting
  deck-shape change.
- Audit: first prove the action can trigger and targets remain legal.
- Reject: speculative text parsing or action-neutral implementation.

## Ordering and copy effects

**Main Mult ordering**

- Signal: main-stage XMult resolves before later main-stage additive Mult.
- Change: repair one dependency inversion per observation.
- Guard: exclude scored-card, held-card, per-card, retrigger, and unresolved
  copy effects.

**Copy dependency graph**

- Signal: Blueprint or Brainstorm has a modeled target whose copied value
  changes with the current hand.
- Change: evaluate explicit source-to-target dependencies and legal moves.
- Audit: count actual copy states and changed targets before spending Episodes.
- Reject: multiple-copy chains are unresolved or the experiment is nearly
  behavior-neutral.

## Bosses and late-game horizon

**Boss constraint planner**

- Signal: legal but zero-value hands, forbidden repeats, one-hand limits, or
  face-down information.
- Change: filter candidates by public Boss rules before scoring.
- Guard: express constraints semantically; avoid name-only patches when the
  rule text or structured parameters suffice.

**Threat-ratio spending**

- Signal: the build survives mid-game but lacks projected Ante 7–8 throughput.
- Change: compare conservative scoring capacity with upcoming Blind pressure
  and increase purchase/reroll urgency as the ratio falls.
- Audit: inspect whether changed spending occurs before late deaths, not after
  the build is already doomed.
- Guard: preserve early interest when survival capacity is comfortably above
  pressure.

## Avoid low-value experiments

Do not spend environment budget on:

- a feature that changes no replay decision;
- a more exact score component without testing the downstream selector;
- one global threshold tuned from a single index;
- a named-Joker bonus that duplicates structured role modeling;
- generic Joker sorting across unresolved copy dependencies;
- a broad rule whose gain comes with large early regressions;
- confirmation replay used to create another patch;
- a module refactor bundled with a strategic behavior change;
- several capabilities combined in one candidate.
