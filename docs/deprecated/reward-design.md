# Reward Design

## Goal

Reward should measure whether a patch improves the heuristic-learning system,
not whether it exploits one seed or bloats the policy.

## Single-Step Reward

A first version can use:

```text
R_step =
  w_train * normalized_train_delta
+ w_val * normalized_validation_delta
+ w_heldout_probe * normalized_probe_delta
- w_regression * regression_count
- w_invalid * invalid_patch
- w_violation * evaluator_violation
- w_complexity * complexity_delta
- w_runtime * runtime_delta
- w_variance * score_variance_increase
```

Held-out probe scores may be used internally for reward, but detailed held-out
failure traces should not be shown to the sampler.

## Learning-Curve Reward

For multi-step experiments, optimize the whole improvement trajectory:

```text
R_curve =
  area_under_heldout_learning_curve
+ final_heldout_score
- cumulative_rollout_cost
- cumulative_regressions
- cumulative_complexity_growth
```

This is closer to the actual research question.

## Candidate Selection Reward

For best-of-k sampling:

1. Run cheap train rollouts for all candidates.
2. Filter invalid and violating patches.
3. Run validation rollouts for top candidates.
4. Select using validation score, regression count, and complexity.
5. Report final metrics on held-out seeds.

## Reward Components

### Performance

- mean score delta;
- median score delta;
- worst-quartile improvement;
- success rate;
- task-specific normalized objective.

### Robustness

- held-out seed performance;
- environment variant performance;
- disturbance recovery;
- regression count.

### Reliability

- patch applies cleanly;
- policy imports or compiles;
- no runtime exceptions;
- no timeout;
- no sandbox violation.

### Maintainability

- lines of code delta;
- number of branches;
- duplicate code estimate;
- runtime cost;
- memory use.

Maintainability penalties should be small enough not to suppress real
improvements, but large enough to prevent benchmark-specific rule explosions.

## Anti-Hacking Rules

Assign a hard negative reward if sampled policy code:

- imports rollout internals;
- reads hidden seed lists or held-out artifacts;
- writes outside the workspace;
- uses wall-clock or filesystem tricks to infer evaluation mode;
- disables safety checks;
- modifies evaluator files;
- exceeds resource limits.

## Reporting

Always report:

- raw task score;
- normalized reward;
- train score;
- validation score;
- held-out score;
- invalid patch rate;
- regression rate;
- average rollout cost;
- code complexity trend.
