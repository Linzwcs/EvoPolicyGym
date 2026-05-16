# Research Plan

## Central Question

Can training improve a model's ability to perform heuristic learning through
iterative code patches and rollout feedback?

The target ability is not direct task solving. The target ability is to update
an external heuristic system over multiple evaluation rounds.

## Framing

We model heuristic learning as an outer-loop Markov decision process. See
[`hl-model.md`](hl-model.md) for the full state, action, transition, reward, and
training-data formulation.

```text
outer state H_t:
  policy code
  rollout summaries
  failure clusters
  regression history
  code complexity signals
  previous patch outcomes

outer action A_t:
  patch to the heuristic system
  optional diagnosis and risk statement

transition:
  apply patch
  run static checks
  run rollout evaluation
  summarize failures

reward:
  improvement in train and held-out performance
  minus regressions, invalid patches, excess complexity, and rollout cost
```

## Hypotheses

### H1: Training improves learning efficiency

Under the same rollout budget, a trained sampler reaches higher held-out score
than the base model.

Primary metric:

```text
AUC of held-out score over rollout budget
```

### H2: Training reduces destructive edits

A trained sampler produces fewer invalid patches, compile failures, evaluator
violations, and regressions against prior seeds.

Primary metrics:

- invalid patch rate;
- static check failure rate;
- train regression count;
- held-out regression count;
- policy runtime failures.

### H3: Training improves failure-to-patch alignment

A trained sampler should make edits that directly address observed failure
modes instead of adding unrelated code.

Primary metrics:

- failure cluster coverage;
- score gain on affected seeds;
- regression-free improvement rate;
- human audit on a sample of patch rationales.

### H4: Training transfers across task families

Improvement should not be limited to the exact environments used for sampling.

Evaluation splits:

- held-out seeds;
- held-out environment variants;
- held-out tasks in the same family;
- held-out task families.

## Baselines

- Base code model sampler with the same prompt and budget.
- Base model with best-of-k rollout reranking.
- Random patch or random parameter search where applicable.
- Evolutionary search over hand-designed heuristic templates.
- SWE-RL-style single-step code repair objective, if a comparable model is
  available.
- Human-written heuristic baseline for simple tasks.

## Success Criteria

The minimum credible result is a trained sampler that improves learning
efficiency on held-out seeds and at least one held-out task variant.

The stronger result is cross-family transfer: training on cheap control,
scheduling, and numerical tasks improves heuristic-learning behavior on a new
task family.

## Main Risks

- The model memorizes benchmark-specific tricks.
- Reward overfits train seeds.
- Summaries hide the true failure cause.
- Rollout cost makes online RFT impractical.
- Policy code learns to exploit evaluator details.
- Multi-step credit assignment is too noisy for simple single-step reward.

## Risk Controls

- Keep train and held-out seeds separate.
- Hide held-out failure traces from the sampler.
- Hash and archive every task spec, policy file, prompt, and rollout config.
- Penalize invalid patches and evaluator violations heavily.
- Track code size and runtime cost.
- Compare single-step reward with learning-curve reward.
