# Layered modeling stack

Diagnose and optimize the Policy as a stack of contracts. Fix the lowest
incorrect layer that explains the observed decision; do not compensate for a
mechanics or prediction defect with a higher-level heuristic.

## Contents

- The model layers
- Data contracts between layers
- Diagnose the failing layer
- Optimize one layer
- Propagate uncertainty
- Combine objectives
- Migrate an existing heuristic Policy

## The model layers

### L0: observation and legality

Convert raw public dictionaries into a normalized `StateView` and
`LegalCatalog`.

- Exact responsibilities: presence, types, defaults, unknown values, stable
  identity, ephemeral indices, and legal descriptors.
- Failure evidence: exceptions, stale indices, false equality between hidden
  values, or invalid Actions downstream.
- Tests: normalization tables and exact legal-action fixtures.

### L1: mechanics and effect semantics

Represent public game rules without deciding what is strategically valuable.

- Outputs: hand classification, scoring phases, `EffectProfile`, card
  modifiers, Boss constraints, and action consequences that are exact.
- Separate scored-card, held-card, retrigger, main additive, main
  multiplicative, economy, scaling, and copy dependencies.
- Failure evidence: predicted/actual rule disagreement or an impossible hand
  classification.
- Tests: pure table-driven mechanics and `last_hand` component checks.

### L2: Episode plan and build abstraction

Summarize the public run into stable Episode-local strategic context.

- Outputs: primary/secondary hand plans, covered roles, scaling trajectory,
  economy phase, switching cost, active constraints, and confidence.
- Do not store future draws, hidden Case identity, or ephemeral list indices.
- Failure evidence: hand-plan oscillation, stale shop intent, redundant roles,
  or state leaking across reset boundaries.
- Tests: sequential state transitions and explicit reset scopes.

### L3: action outcome prediction

Predict the consequences of each legal semantic intent, conditional on L0–L2.

- Play: score range, lethal probability, cards consumed, future hands.
- Discard: draw distribution, outs, retained activation, remaining discards.
- Shop: money/interest delta, slot delta, role delta, scaling delta.
- Pack/consumable: target legality, deck-shape delta, permanent growth.
- Reorder: resolution dependency and estimated score delta.
- Always return uncertainty or confidence with the estimate.
- Failure evidence: correct mechanics but systematic outcome error, especially
  against `last_hand` or observed draw/shop transitions.

### L4: action value and horizon

Translate predicted outcomes into comparable strategic components:

```text
ActionValue(
  immediate_survival,
  blind_progress,
  build_growth,
  economy,
  switching_cost,
  late_game_ceiling,
  uncertainty,
)
```

Keep components separate until the final comparison. Failure evidence includes
correct immediate predictions but harmful play/discard, purchase, or reroll
choices over the remaining Blind horizon.

### L5: policy and intent selection

Apply Boss constraints, risk posture, remaining hands/discards, Ante pressure,
and the Episode objective to rank intents.

- Preserve a high-confidence lethal fast path.
- Prefer constrained or lexicographic comparison for release-blocking legality
  and imminent survival before softer growth/economy tradeoffs.
- Use deterministic tie-breaking.
- Failure evidence: values are reasonable but the wrong intent wins because of
  priority, threshold, or fallback logic.

### L6: Action execution

Translate the selected intent through the current legal descriptor.

- Emit exact fields and current indices only.
- Revalidate multi-observation transactions.
- Fall back to another currently legal intent; never repair an invalid Action.
- Failure evidence: any Policy failure, stale target, malformed payload, or
  descriptor mismatch.

## Data contracts between layers

Use explicit immutable values where practical:

```text
raw observation
  -> StateView + LegalCatalog
  -> MechanicsSnapshot + EpisodePlan
  -> OutcomeEstimate(intent)
  -> ActionValue
  -> ranked Intent
  -> admitted Action
```

Do not let L4 or L5 parse raw rule dictionaries. Do not let L1 contain shop
tier weights. Do not let L6 decide strategy. A layer may depend only on public
values from lower layers plus documented Episode-local state.

## Diagnose the failing layer

Find the earliest layer whose output becomes wrong:

| Symptom | First layer to inspect |
|---|---|
| Invalid Action or stale target | L0 or L6 |
| Wrong hand type or Boss legality | L1 |
| Correct rules, wrong score/draw/shop consequence | L3 |
| Good prediction, bad build summary | L2 |
| Good prediction, bad long-term tradeoff | L4 |
| Good values, wrong threshold/fallback | L5 |
| Good local choices, weak late ceiling | L2 and L4 |
| Same public sequence behaves differently | L2 reset or nondeterminism |

Inspect a replay decision by recording layer outputs in tests or analysis
helpers, never by adding Host I/O or persistent logging to `act()`.

## Optimize one layer

1. Freeze the layer's input/output contract.
2. Add a local invariant or calibration test.
3. Implement the smallest correction inside that layer.
4. Run lower-layer tests to prove the input facts remain valid.
5. Run every downstream selector and Action test because decisions may move.
6. Count eligible states, changed outcomes, changed values, and changed intents
   separately.
7. Run matched indexed A/B only after local propagation is understood.

When a lower-layer fix exposes a bad upper-layer threshold, prefer two
experiments: first the semantic correction, then the policy response. Combine
them only when the old threshold is nonsensical under the corrected units and
the dependency is explicit.

## Propagate uncertainty

- Mark exact, estimated, and unknown contributions separately.
- Widen score or value ranges when effects are unmodeled.
- Discount speculative build growth rather than pretending it is zero.
- Prefer robust intents when close estimates overlap materially.
- Preserve high-confidence opportunities; do not let one unknown effect make
  every action indistinguishable.
- Test confidence thresholds on development indices, never frozen
  confirmation replay.

Calibration metrics must match the layer. Use score error and ranking accuracy
for L3, not Episode wins. Use matched action changes and survival for L4–L5,
not only lower numerical error.

## Combine objectives

Do not hide every concern inside one undocumented scalar. First enforce:

1. exact legality and no Policy failure;
2. public Boss and action constraints;
3. imminent survival when failure is otherwise near-certain.

Then compare expected Episode reward using separately observable survival,
progress, growth, economy, ceiling, and uncertainty terms. Document units and
tie-breaking. Report which component caused a decision to change.

## Migrate an existing heuristic Policy

Do not rewrite everything at once:

1. Wrap raw observation access in L0 without changing behavior.
2. Extract exact hand/effect mechanics into L1.
3. Introduce `EpisodePlan` fields only for state already used by decisions.
4. Replace one heuristic with an L3 outcome and L4 value estimate.
5. Keep the old rule as a conservative fallback outside high-confidence
   coverage.
6. Expand coverage only after replay and matched evidence.
7. Remove the fallback after the new path covers its states and remains safe.

This staged migration keeps each Program digest interpretable and revertible.
