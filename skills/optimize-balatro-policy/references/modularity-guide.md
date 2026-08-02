# Modular Policy architecture

Use modules to give every Policy responsibility one owner and one dependency
direction. Use `modeling-stack.md` separately to diagnose which reasoning layer
is wrong. A model layer describes the decision contract; a module owns the code
that implements one or more cohesive contracts.

## Contents

- Separate model layers from code modules
- Assign one responsibility to one owner
- Keep dependencies acyclic
- Define narrow contracts
- Isolate Episode-local state
- Decide when to split or merge
- Refactor without changing strategy
- Test module boundaries
- Migrate a monolithic Policy
- Review the resulting architecture

## Separate model layers from code modules

Do not require one file per model layer. That creates tiny pass-through modules
and makes a single capability expensive to trace. Instead:

- use the layers to locate the earliest incorrect output;
- use modules to assign ownership of stable concepts;
- let one cohesive module implement adjacent contracts when they change
  together;
- split a module when its responsibilities have different dependencies, reset
  scopes, tests, or reasons to change.

For example, `hands.py` may own exact hand classification from L1 and candidate
enumeration used by L3. Do not put build-aware action ranking there merely
because the final action is a hand play.

## Assign one responsibility to one owner

Use this as a responsibility map, not a mandatory file list:

```text
policy.py          ABI adapter and composition
policy_system/
  state.py         observation normalization and EpisodePlan lifecycle
  actions.py       LegalCatalog, ActionGateway, and exact payload construction
  hands.py         hand classification and legal card combinations
  effects.py       visible effect semantics and resolution phases
  scoring.py       play outcome estimates and calibration
  draws.py         outs, draw distributions, and discard outcomes
  build.py         role coverage, synergy, scaling, and replacement value
  economy.py       reserves, interest, purchases, rerolls, packs, and skips
  constraints.py   Boss and other public action/outcome constraints
  consumables.py   target generation and deck-shape consequences
  planning.py      horizon comparison and ActionValue
  strategy.py      candidate orchestration and final intent ranking
tests/
  ...              pure, boundary, transition, and replay tests
```

Merge adjacent small owners when the Program is compact. For example, merge
`build.py` and `economy.py` until each has substantial independent behavior.
Split them once one can change without understanding the other. Preserve these
ownership rules even when filenames differ:

- `state` owns raw observation interpretation and reset scopes;
- mechanics modules own exact public game semantics;
- prediction modules own consequences, ranges, and confidence;
- planning owns cross-action value and horizon;
- strategy orchestrates existing owners but does not reimplement them;
- actions owns the final conversion from semantic intent to legal Action.

One responsibility, one owner. Do not maintain parallel hand scorers, effect
parsers, shop valuations, legal-action builders, or Joker role tables.

## Keep dependencies acyclic

Direct dependencies toward facts and pure models:

```text
policy -> strategy
strategy -> planning + state + actions
planning -> scoring + draws + build + economy + constraints + consumables
scoring/draws/build/economy/constraints/consumables -> hands + effects + values
hands/effects/values -> normalized public domain values
actions -> LegalCatalog + intents + normalized identity
state -> normalized public domain values
```

Adapt the graph to the actual Program while enforcing:

- lower mechanics modules never import `strategy` or `policy`;
- `state` never selects or emits an Action;
- `actions` never decides whether buying, discarding, or playing is valuable;
- `scoring` never parses a raw observation dictionary;
- `effects` never imports build weights, economy thresholds, or strategy;
- sibling planners exchange explicit values through their caller instead of
  importing each other in a cycle;
- `policy.py` performs no domain computation.

If two modules need each other, move the shared value or pure fact into a
lower neutral owner. Do not hide cycles behind runtime imports, callbacks, a
service locator, or an untyped `context` dictionary.

## Define narrow contracts

Pass immutable, typed values where practical. Prefer:

```text
StateView + LegalCatalog + EpisodePlan
  -> tuple[IntentCandidate, ...]
  -> ActionValue per candidate
  -> ranked Intent
  -> admitted Action
```

Keep contracts smaller than the raw observation:

- pass the hand, visible cards, and modeled effects to scoring;
- pass cash, prices, reserve, slots, and build deltas to economy;
- pass public constraints to candidate filtering;
- pass stable public identity in a staged intent, never an ephemeral index.

Do not leak raw dictionaries past the observation adapter. Do not return
unchecked Action dictionaries from planners. Do not use a generic `utils.py`
or `helpers.py` as an owner for domain behavior. Name a shared module after the
concept it owns, such as `values.py`, `identity.py`, or `probability.py`.

Document units and uncertainty at module seams. For example, distinguish an
estimated in-game score range from expected Episode reward and distinguish
exact mechanics from approximate effect coverage.

## Isolate Episode-local state

Prefer a small stateful shell around a pure decision core:

- keep Policy instance lifecycle and composition in `policy.py`;
- keep `EpisodePlan`, transition detection, and pending transaction state in
  the state owner;
- keep mechanics, predictions, and value calculations pure;
- let strategy read and request state transitions through explicit methods;
- let the Action gateway resolve stable intent identities to current indices.

Give every mutable field one writer and a documented reset boundary. Avoid
module globals and cross-Episode caches. Cache pure derived facts only when
their public identity is stable, and never cache observation indices across
calls.

## Decide when to split or merge

Split a responsibility when at least one of these holds:

- two callers need the same behavior and currently duplicate it;
- the behavior has an independent invariant or calibration test;
- its dependencies point lower than the rest of the current module;
- it has a distinct mutation or reset scope;
- changing it should not require understanding unrelated phases;
- a strategy branch is re-parsing, rescoring, or reconstructing Actions.

Do not use line count alone as the split criterion. Keep code together when it
shares one invariant, one change reason, and one dependency set.

Merge or remove a module when it only forwards every argument, contains a
single incidental helper, creates an import cycle, or gives a second name to
an existing owner. Avoid one-class-per-file ceremony, abstract base classes
with one implementation, plugin frameworks inside `program/`, and
version-suffixed replacements such as `scoring_v2.py`.

## Refactor without changing strategy

Refactor and strategy change are separate experiments.

1. Inventory existing owners, raw-field access, Action construction, mutable
   state, duplicated constants, and import edges.
2. Choose one leaf responsibility and define its input/output contract.
3. Add characterization tests from authorized public fixtures before moving
   code. Record control decisions, not only final Actions when useful.
4. Extract the responsibility without changing thresholds, tie-breaking,
   fallback order, numeric units, or state reset behavior.
5. Route all callers to the new owner and delete the duplicate implementation.
6. Run pure, boundary, transition, and sequential replay tests.
7. Require identical intents and Actions on the refactor replay corpus. Treat
   any difference as unexplained until classified.
8. Use a small matched same-index control/refactor submission only when local
   equivalence cannot cover runtime behavior. Expect score neutrality.
9. Start a new digest and experiment for the subsequent strategic change.

Do not claim a reward improvement from modularization. Its evidence is lower
defect risk, observable contracts, action equivalence, and faster isolated
experiments. If a refactor intentionally changes decisions, name the changed
capability and evaluate it as a strategic experiment.

## Test module boundaries

Mirror the dependency graph in tests:

- test normalized values without strategy;
- test hand and effect mechanics with tables;
- test predictions with fixed model inputs and confidence;
- test value comparison with fixed outcome estimates;
- test state transitions as public observation sequences;
- test intents independently from Action translation;
- test the Action gateway against exact legal descriptors;
- test the composed strategy on persistent public replay fixtures.

Add contract tests whenever multiple modules consume the same value. Assert
deterministic ordering and tie-breaking explicitly. Keep fixtures minimal and
authorized; never encode hidden Case or seed identity.

For a refactor, compare at the earliest observable seam that moved and every
downstream seam. Identical final Actions alone can conceal a corrupted
prediction that happens to preserve current ranking.

## Migrate a monolithic Policy

Extract stable leaves before the orchestrator:

1. centralize observation normalization and legal descriptors;
2. centralize exact Action construction and admission;
3. extract hand classification and structured effect semantics;
4. extract scoring and draw outcome prediction;
5. isolate EpisodePlan state and transition resets;
6. extract build, economy, consumable, and constraint models as evidence
   justifies them;
7. make planning compare shared `ActionValue` components;
8. reduce strategy to candidate orchestration, ranking, and safe fallback;
9. reduce `policy.py` to ABI construction and delegation.

Keep the old path as a temporary characterized fallback only when the new
module does not yet cover all public states. Give that fallback an explicit
removal condition. Never keep two authoritative models indefinitely.

## Review the resulting architecture

Before submission, verify:

- Can each behavior be named with one owning module?
- Does the import graph follow one direction without runtime-cycle tricks?
- Is raw public data normalized exactly once?
- Is every Action constructed and admitted in one gateway?
- Does every mutable field have one writer and reset scope?
- Do planners return semantic intents rather than payload dictionaries?
- Are mechanics, prediction, value, and selection separately testable?
- Are thresholds and units owned near the model that interprets them?
- Did the refactor preserve intent and Action behavior on replay?
- Is any remaining duplicate model temporary with a removal condition?

Prefer the smallest architecture that satisfies these checks. Modularity is a
means to improve experiment quality and Policy reliability, not an objective
that competes with complete-run reward.
