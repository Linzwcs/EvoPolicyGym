---
name: optimize-balatro-policy
description: Improve and validate EvoPolicyGym Policy Programs for the Jackdaw Balatro Benchmark. Use when editing program/policy.py, analyzing feedback or replay.jsonl, allocating the Episode budget, diagnosing invalid actions or Policy failures, comparing submissions under RNG, or selecting a final Balatro submission.
---

# Optimize Balatro Policy

Build a robust, observation-driven player that can win complete runs. Prefer a
small coherent decision model over an expanding catalog of encounter-specific
rules.

## Respect the evidence boundary

- Treat the Benchmark specification, the current observation, and
  `legal_actions` as authoritative.
- Use only public feedback. Never infer a hidden seed, future draw, shop roll,
  or private Case identity.
- Edit only `program/`. Never modify `feedback/`.
- Re-read indices after every action. Entity and hand indices are ephemeral.

## Run the improvement loop

1. Inspect `program/policy.py`, `feedback/latest.json`, its Feedback document,
   and every advertised Artifact before changing the Policy.
2. Submit the unchanged Program to establish a baseline.
3. Keep an evidence ledger containing submission ID, Program digest, Episode
   count, score, failures, win count, Ante reached, and the hypothesis tested.
4. Change one decision principle at a time. Do not add a named-object special
   case merely because it appeared in one losing replay.
5. Re-submit unchanged promising candidates across more Episodes. Treat a
   four-Episode score as a noisy observation, not a ranking.
6. Reserve enough budget for final validation. Select the candidate supported
   by the most relevant evidence, not the largest observed small-sample score.
7. Call `evopolicygym finish SUBMISSION_ID` successfully before exiting.

When the submission cap permits it, use small batches for correctness and
larger batches for selection. If batches receive different hidden Cases,
compare distributions and repeated identical digests rather than attributing
every score change to the latest edit.

## Build the Policy in layers

Keep the layers separate so that failures are diagnosable:

1. **Safety:** Normalize nullable visible fields, validate phase assumptions,
   and construct exact action shapes from `legal_actions`.
2. **Tactics:** Enumerate legal card selections and estimate their immediate
   Chips, Mult, conditional effects, and card order.
3. **Draw planning:** Use visible deck counts to compare playing now with the
   expected value of discarding. Count outs; do not chase a five-card hand only
   because it is categorically stronger.
4. **Build planning:** Track the run's primary hand plan and its Chips, additive
   Mult, X Mult, retrigger, economy, and deck-shaping engines.
5. **Economy:** Compare buy, sell, pack, Voucher, reroll, interest, and
   next-round choices by marginal value to the current build.
6. **Long-horizon control:** Check whether the engine's observed growth can
   meet upcoming Blind scaling. Surviving the next Blind is not enough.

Use visible `rule.summary`, `rule.parameters`, mutable `ability` values,
`poker_hands`, `deck`, `last_hand`, and Blind rules. A purchase ranking and a
hand scorer must agree: do not value a Joker highly if the Policy cannot model
or activate its effect.

## Establish a build

- Cover base Chips, additive Mult, and multiplicative growth; five unrelated
  additive Jokers are not a late-game engine.
- During early Antes, buy enough immediate scoring to survive while preserving
  an economy. Then concentrate Planets, deck changes, and Joker choices around
  one or two compatible hand types.
- Re-evaluate the plan when a genuinely stronger engine appears, but charge a
  switching cost for abandoned hand levels and deck shaping.
- Model temporary, scaling, conditional, and consumable effects differently.
  Do not label a decaying early-game Joker a permanent power engine.
- Evaluate Joker order and played-card order wherever resolution order matters.
- Inspect the active Boss rule explicitly. A generic target-score check is not
  sufficient Boss handling.

## Harden every action path

- Assume visible numeric fields may be `null` when a card is face-down. Use
  explicit normalization such as `value or 0`; a default passed to `.get()`
  does not replace an existing `None`.
- Exercise shop, pack, consumable, face-down hand, full-slot, zero-money,
  no-discard, and Boss states from replay observations.
- Replay the Policy locally on the exact last public observation when a Policy
  fails. Fix the underlying invariant across all phases.
- Never repair an invalid action by guessing. If a desired action is absent
  from `legal_actions`, choose a different legal plan.

## Avoid these failure modes

- Do not optimize the current hand while ignoring the remaining deck.
- Do not spread Planet levels across whichever hand happened to appear last.
- Do not hard-code an ever-growing Joker tier list without state-dependent
  marginal scoring.
- Do not bundle several strategic changes into one noisy submission.
- Do not choose the historical maximum score without accounting for Episode
  count, failures, and variance.
- Do not claim stability after testing only the action paths seen so far.

The objective has a large win bonus. Use cleared Blinds as progress evidence,
but optimize the strategy for defeating the Ante 8 Boss rather than maximizing
short-run Blind count alone.
