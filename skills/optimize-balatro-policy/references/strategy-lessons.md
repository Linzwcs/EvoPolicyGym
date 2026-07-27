# Reusable Balatro strategy lessons

Apply these lessons as model invariants, not as a list of named encounter
patches.

## Couple scoring to the decision horizon

- A more exact immediate hand score can make a myopic selector worse. Whenever
  scoring changes, test the play/discard choice and remaining-hands horizon,
  not calibration alone.
- Separate base Chips/Mult, scored-card triggers, held-card triggers, main
  additive Mult, and main XMult into resolution phases. Reorder only effects
  whose phase and dependency are modeled.
- Use `last_hand` to find systematic prediction error on development replays.
  Prioritize effects that occur often enough to change decisions.
- Treat unknown effects as uncertainty. Do not silently score them as zero.

## Preserve information boundaries

- Face-down cards with missing rank or suit are unknown, not equal. Never form
  a Pair, Flush group, or deterministic out from shared `None` values.
- Draw-pile composition supports probabilities, not future-card prediction.
- Keep all indices ephemeral and rebuild intents from the current observation.

## Optimize draw throughput safely

- Compare discard value with playing a nonlethal hand when extra played cards
  can cycle weak kickers and improve the next draw.
- Add kickers only when they are legal, unmodified low-value junk, outside the
  scoring hand and valuable draws, and enough future hands remain.
- Guard throughput changes by Ante, remaining hands/discards, Blind deficit,
  Boss rules, and score confidence. Broad early activation creates severe
  regressions.

## Plan the build and shop jointly

- Value a Joker by marginal survival and growth in the current build, not by a
  standalone tier. Include role coverage, activation rate, scaling horizon,
  slot opportunity cost, sell proceeds, interest, and replacement sequence.
- When slots are full and cash is safely above reserve, a bounded paid reroll
  can search for upgrades. Record rerolls per shop and cap them.
- A purchase rule that never triggers is not an optimization. Count eligible
  shops and changed decisions before evaluating it.
- Conservative Planet and consumable use is safer than speculative targets,
  but behavior-neutral safety code is not evidence of strength.

## Handle Joker order as a dependency problem

- Put main-stage additive Mult before main-stage XMult when no other dependency
  overrides that order.
- Treat Blueprint, Brainstorm, retriggers, per-card effects, and multi-copy
  chains as explicit dependency graphs. Do not apply a generic sort while copy
  semantics are unresolved.
- Move at most one adjacent inversion per observation unless action semantics
  guarantee an atomic full reorder.

## Learn correctly from failures

- Diagnose only from submissions assigned to baseline, diagnosis, or capability
  development. Frozen-confirmation regression tells you to reject or reorder a
  shortlist, not how to patch.
- Prefer a narrow guarded capability that holds up across several fresh
  submissions over a broad rule with a larger pooled mean and catastrophic
  early regressions.
- Separate engineering correctness, average survival, late-tail reach, and win
  rate. A change may improve one without demonstrating another.
