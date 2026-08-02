# Implementation playbook

Implement one capability through stable Policy-system seams. Favor code that
can be tested without launching the environment and reverted without
reconstructing unrelated behavior.

## Contents

- Trace before editing
- Localize the model layer
- Preserve module ownership
- Add a capability in layers
- Normalize observations
- Model Episode-local state
- Separate intents from Actions
- Implement multi-observation transactions
- Structure scoring and ordering
- Test the change
- Control scope and performance
- Review before submission

## Trace before editing

For the target decision, trace:

```text
raw observation
  -> normalized StateView
  -> mechanics and EpisodePlan
  -> action outcome estimate
  -> action value and horizon comparison
  -> ranked intent
  -> ActionGateway
  -> exact Action
```

Identify the narrowest missing seam. Do not add a second raw-dictionary parser,
shop tier list, scoring model, or Action constructor to bypass existing code.
Run current unit tests and record the control digest before editing.

## Localize the model layer

Use `modeling-stack.md` to find the earliest incorrect output. Inspect lower
layers before changing an upper one:

- normalize missing or hidden data in L0;
- encode exact public semantics in L1;
- repair Episode-local abstraction and resets in L2;
- calibrate score, draw, shop, or reorder consequences in L3;
- change survival/growth/economy tradeoffs in L4;
- adjust priority, threshold, and fallback behavior in L5;
- repair exact Action translation in L6.

Keep these outputs observable in pure tests. Never patch L5 weights to conceal
an L1 or L3 defect.

## Preserve module ownership

Use `modularity-guide.md` before extracting, splitting, merging, or rewiring
Policy modules. Treat model layers and code modules as different views:
layers locate an incorrect contract; modules assign ownership and dependency
direction.

- Give raw normalization, mechanics, prediction, value, strategy, and Action
  construction one authoritative owner each.
- Direct imports toward normalized facts and pure lower models. Never make a
  mechanics module depend on strategy or let sibling planners form a cycle.
- Extract one leaf responsibility at a time and characterize its current
  intent and Action behavior before moving it.
- Delete the superseded implementation after all callers use the new owner.
- Keep a modular refactor and a strategic behavior change in separate digests
  and experiments.

Do not introduce generic utility buckets, one-function pass-through modules,
abstract frameworks with one implementation, or version-suffixed parallel
models. Merge adjacent small modules when they share one invariant, dependency
set, and reason to change.

## Add a capability in layers

Use this patch order:

1. Identify and freeze the affected layer contract.
2. Add an invariant or table-driven test for its missing behavior.
3. Add or extend the pure function in the lowest incorrect layer.
4. Integrate the changed output through every dependent upper layer.
5. Route the resulting intent through the existing Action gateway.
6. Add state-transition tests if memory spans observations.
7. Run a replay opportunity audit and inspect output/value/action changes.
8. Run matched environment evidence only after local gates pass.

Keep the pure function independent of Policy process I/O. Pass a small frozen
context value instead of the whole raw observation when practical.

## Normalize observations

- Centralize nullable mappings, finite numeric conversion, rule summaries,
  structured parameters, card identity, and resource defaults.
- Preserve the distinction between absent, unknown, zero, and false. In
  particular, two unknown ranks or suits are not equal.
- Treat observation indices as ephemeral. Resolve them again on every `act()`.
- Use stable public card keys only for intent identity; never store a temporary
  list index as cross-observation identity.
- Represent unmodeled public effects explicitly with confidence or an unknown
  role instead of inventing zero behavior.

## Model Episode-local state

Use a small `EpisodePlan` state machine. Give every field a documented reset
scope:

| State | Reset boundary |
|---|---|
| primary/secondary hand plan | Episode or explicit strategy switch |
| round discard counters | new Blind/round |
| rerolls in current shop | new shop |
| pending sell/purchase intent | completion, illegality, or shop change |
| observed score calibration | Episode-local only if used by decisions |

Derive resets from public observation transitions. A new Policy instance must
start every Episode; do not build cross-Episode learning into `act()`.

## Separate intents from Actions

Make planners return semantic intents such as:

```text
Play(cards)
Discard(cards)
Buy(stable_card_key)
Sell(stable_owned_key)
UseConsumable(stable_key, target_keys)
Reorder(source_key, destination)
```

Translate an intent only after reading the current legal descriptor. The
gateway must:

- emit exactly the allowed fields;
- check the action kind is currently present;
- check target and card-index membership;
- enforce uniqueness, ordering, and min/max cardinality;
- reject stale intent instead of repairing it silently;
- rank a different legal intent or use a safe legal fallback.

Never pass descriptor metadata back as Action data.

## Implement multi-observation transactions

Selling to make room and then buying is not atomic:

1. Choose replacement using current public values.
2. Emit only the legal sell Action.
3. Store a stable offered-card key and expected shop identity.
4. On the next observation, verify the shop, offered card, price, money, slot,
   and current `buy_card` target.
5. Buy only if the original value condition still holds.
6. Clear the intent on success, disappearance, illegality, phase change, or
   conflicting higher-priority survival action.

Test every cancellation branch. Apply the same pattern to other staged actions.

## Structure scoring and ordering

Keep scoring phases explicit:

```text
base hand
  -> scored-card effects and retriggers
  -> held-card effects
  -> main additive Chips/Mult
  -> main multiplicative Mult
```

Follow the public rule semantics when an effect belongs elsewhere. Return both
an estimate and confidence when the model is incomplete.

For Joker movement:

- construct dependencies from visible public rules;
- move only a proven inversion;
- prefer one adjacent legal move per observation;
- disable generic sorting when copy or trigger dependencies are unresolved;
- test the final resolution order, not only the list permutation.

Whenever score calibration changes, rerun hand selection and discard tests.
Numerical accuracy alone is not a Policy improvement.

## Test the change

Maintain four test layers:

1. **Pure mechanics**: hand classification, effects, value, outs, ordering.
2. **Action contract**: exact shape and descriptor admission.
3. **State transitions**: round/shop/Episode reset and staged intents.
4. **Public replay regression**: sequential observations and changed decisions.

Add small fixtures containing only authorized public fields needed by the
invariant. Cover:

- hidden and debuffed cards;
- zero hands or discards;
- full slots and insufficient money;
- disappearing shop inventory;
- packs and targeted consumables;
- Boss restrictions;
- copy and ordering interactions;
- deterministic repeated calls with equivalent state and memory.

Use actual `last_hand` evidence for calibration tests, but do not make replay
fixtures depend on private seed or Case identity.

## Control scope and performance

- Change one decision capability per experiment.
- Prefer extending a shared role or phase model over a named-object exception.
- Keep constants close to the model they parameterize and document units.
- Avoid catch-all exception fallbacks that hide correctness defects.
- Enumerate the complete legal small hand space before adding search pruning.
- Cache only normalized or pure values whose public identity is stable; never
  cache ephemeral indices across observations.
- Keep `act()` free of file, socket, subprocess, credential, and Host access.
- Use only dependencies already available to the submitted Program.

Refactor first when the proposed change would duplicate normalization, action
admission, scoring, or shop valuation. Prove intent and Action equivalence on
replay. Treat the later capability change as a separate experiment before
claiming strategic improvement.

## Review before submission

- Does the patch affect only the stated capability?
- Is every new branch reachable in a public opportunity audit?
- Are unknown and hidden values handled conservatively?
- Can every emitted Action pass the current descriptor exactly?
- Does Episode-local state reset at the right boundary?
- Are transaction intents cleared on every invalidation path?
- Does the score model use the correct resolution phase?
- Are copy dependencies either modeled or explicitly excluded?
- Do tests cover the previous failure and the safe fallback?
- Can the patch be reverted exactly and the digest verified?
- Does every changed responsibility still have one owner and an acyclic import
  path?
