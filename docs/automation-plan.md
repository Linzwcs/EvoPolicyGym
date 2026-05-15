# Automation Plan

## Goal

Run heuristic-learning experiments without human interaction in the inner loop.

The system should use a model as a patch sampler, then let Python orchestration
apply patches, run tests, execute rollouts, compute rewards, and log training
transitions.

## Key Decision

Use the model API as the automation boundary.

Do not drive an interactive coding session as the primary experiment mechanism.
The orchestrator should own state, retries, sandboxing, acceptance, and logging.

For experiments that specifically measure Codex as the harness, use
non-interactive `codex exec` as a bounded worker launched by the orchestrator.
See `docs/codex-harness.md`.

```text
orchestrator.py
  -> build prompt state
  -> call model API
  -> receive structured patch proposal
  -> apply patch in isolated workspace
  -> run static checks
  -> run rollout
  -> compute reward
  -> append transition record
```

## Control Loop

```text
H_t:
  policy.py
  rollout summary
  failures
  previous accepted patches

sample k proposals:
  model(H_t) -> patch_1 ... patch_k

evaluate:
  apply patch_i
  compile/import policy
  run cheap rollout
  compute reward_i

select:
  reject invalid or violating patches
  rerank by validation reward
  accept best candidate if it clears threshold

advance:
  H_t -> H_{t+1}
  log all candidates, not only accepted candidates
```

## Process Boundaries

### Sampler

The sampler only proposes structured patch candidates.

Inputs:

- task spec;
- allowed file list;
- current policy code;
- train rollout summary;
- selected train failures;
- prior patch outcomes.

Output:

- diagnosis;
- edit plan;
- unified diff patch;
- expected effect;
- risk.

### Orchestrator

The orchestrator is deterministic experiment control code.

Responsibilities:

- create candidate workspaces;
- call the sampler;
- validate schema;
- apply patch;
- run static checks;
- launch rollouts;
- compute rewards;
- choose accepted patch;
- write `transition.jsonl`.

### Rollout Worker

The rollout worker runs untrusted candidate policy code in isolation.

Responsibilities:

- enforce timeout and resource limits;
- block hidden-state and evaluator access;
- run train, validation, and held-out seed splits;
- write `trials.jsonl`, `failures.jsonl`, and `summary.json`.

## Minimal Python Modules

```text
src/hlrft/
  sampler.py          # model API wrapper
  patching.py         # apply and validate diffs
  rollout.py          # rollout subprocess interface
  reward.py           # scalar and component rewards
  orchestrator.py     # experiment loop
  artifacts.py        # run directory IO
```

## API Sampling

The model call should use structured output so automation does not depend on
parsing free-form text.

Pseudo-code:

```python
proposal = sample_patch(
    model=model_name,
    task_spec=task_spec,
    allowed_files=["policy.py"],
    policy_py=current_policy,
    rollout_summary=summary,
    recent_failures=failures,
)
```

The orchestrator should treat malformed output as an invalid sample and assign
a negative reward.

## Candidate Evaluation

Use staged evaluation to reduce cost.

```text
stage 1: schema validation
stage 2: patch applies cleanly
stage 3: import/compile policy
stage 4: short train rollout
stage 5: validation rollout for top candidates
stage 6: held-out rollout for accepted checkpoints
```

Held-out failure traces should not be added to the next sampler prompt.

## Acceptance Policy

An accepted patch must:

- apply cleanly;
- pass static checks;
- avoid sandbox violations;
- improve validation reward or reduce regressions meaningfully;
- keep complexity within the configured budget.

If no candidate clears the threshold, keep `H_t` and log a no-op transition.

## Training Data

Log every candidate:

```json
{
  "state": {
    "policy_before_sha": "...",
    "rollout_summary_before": {},
    "recent_failures": []
  },
  "action": {
    "diagnosis": "...",
    "patch": "..."
  },
  "result": {
    "patch_applied": true,
    "compile_ok": true,
    "rollout_summary_after": {}
  },
  "reward": {
    "total": 0.0
  }
}
```

This supports:

- SFT from high-quality accepted patches;
- preference training from same-state candidate comparisons;
- offline RL from scored transitions;
- online RFT if the grader can run bounded rollout evaluation.

## Parallelism

The first useful parallelism is candidate-level:

```text
same H_t
  -> sample k patches
  -> evaluate patches in parallel workers
  -> choose one accepted transition
```

Later, run task-level parallelism across seeds and task families.

## Safety

Candidate policy code should run in a restricted subprocess or container.

Minimum controls:

- no network;
- read-only task package;
- write access only to candidate run directory;
- timeout per rollout;
- memory limit;
- explicit allowlist of importable modules;
- hash all evaluator files before and after rollout.

## First Implementation Target

Automate one cheap task end to end:

```text
bin_packing_v0
  initial policy.py
  sample 4 patches per iteration
  run 10 train instances per patch
  validate top 2 candidates
  accept at most 1 patch per iteration
  repeat 10 iterations
```

Success means the system can produce a complete `transition.jsonl` without
manual intervention.
