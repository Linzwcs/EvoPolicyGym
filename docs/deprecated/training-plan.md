# Training Plan

## Stage 0: Protocol Stabilization

Before any model training, stabilize:

- task specs;
- policy interfaces;
- patch format;
- rollout summary schema;
- reward function;
- transition logging.

Exit criterion:

```text
The same policy and seed config produce reproducible summaries and rewards.
```

## Stage 1: Data Collection with Strong Samplers

Use strong code models as proposal samplers.

For each task state:

1. sample `k` patch proposals;
2. apply each patch to an isolated workspace;
3. run static checks;
4. run cheap train rollouts;
5. run validation rollouts for top candidates;
6. log all successes and failures.

Keep invalid patches. They are useful negative examples.

## Stage 2: SFT Baseline

Construct SFT examples from high-quality transitions:

```text
input:
  task spec
  policy before
  rollout summary
  recent failures

target:
  diagnosis
  patch
  risk note
```

Selection rules:

- prefer validation improvement over train-only improvement;
- reject evaluator violations;
- include simple regression-free patches;
- include some simplification patches after high-score runs.

## Stage 3: Preference Training

For each shared state, compare multiple sampled patches.

Preference label:

```text
patch A > patch B
```

when A has better validation reward, fewer regressions, and acceptable
complexity.

This is likely cheaper and more stable than immediate online RFT.

## Stage 4: RFT or External RL

Run online optimization only on cheap tasks first.

The grader should:

- apply the patch;
- run static checks;
- run a bounded rollout;
- compute scalar reward from summary artifacts;
- emit structured diagnostics.

If hosted RFT cannot run the full rollout grader, use external RL:

```text
sample from model
-> run rollout locally
-> update model with offline or batched policy optimization
```

## Stage 5: Transfer Evaluation

Evaluate trained samplers on:

- unseen seeds;
- unseen task variants;
- held-out tasks in known families;
- at least one held-out family.

Use the same rollout budget for all samplers.

## Metrics

Primary:

- held-out learning-curve AUC;
- final held-out score under fixed budget;
- regression-free improvement rate.

Secondary:

- invalid patch rate;
- average number of rollouts to first improvement;
- code complexity growth;
- patch application failure rate;
- average token and rollout cost.

## Decision Gates

Proceed from SFT to preference training only if SFT improves invalid patch rate
or first-improvement latency.

Proceed from preference training to online RFT only if preference training
improves held-out learning-curve AUC on at least two cheap tasks.
