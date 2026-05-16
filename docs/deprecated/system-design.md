# System Design

## Overview

The experiment system has five independent components.

```text
sampler
  -> proposes patch candidates

patch applier
  -> applies one candidate to an isolated workspace

rollout runner
  -> evaluates candidate policy code

summarizer
  -> compresses raw trajectories into failure and metric artifacts

trainer
  -> consumes logged transitions for SFT, preference training, offline RL, or RFT
```

## Data Flow

```text
task_spec + policy.py + rollout_summary + failures
  -> sampler
  -> patch_proposal.json
  -> candidate workspace
  -> static checks
  -> rollout
  -> summary.json + failures.jsonl
  -> reward.json
  -> transition.jsonl
```

## Sampler Responsibilities

The sampler may:

- read the task spec;
- read current policy code;
- read train rollout summaries;
- read selected train failure examples;
- produce a patch and rationale.

The sampler must not:

- read hidden evaluator code;
- read held-out failure details;
- modify rollout runners or reward scripts;
- use network access from policy code;
- write outside the candidate workspace.

## Rollout Runner Responsibilities

The rollout runner should:

- evaluate one policy in isolation;
- enforce time, memory, and file access limits;
- record environment version and seed list;
- emit raw per-trial data;
- produce a machine-readable summary;
- never be modified by sampled patches.

## Summarizer Responsibilities

The summarizer turns raw rollout data into concise model context.

Outputs should include:

- score distribution;
- train and validation deltas;
- representative failures;
- regression list;
- runtime and complexity signals;
- violation flags.

The sampler should usually receive summaries, not raw trajectories.

## Workspace Layout

Each candidate evaluation should use an isolated directory:

```text
runs/<run_id>/
  config.json
  prompt_state.json
  proposal.json
  patch.diff
  workspace/
    policy.py
  rollout/
    trials.jsonl
    failures.jsonl
    summary.json
  reward.json
  transition.json
```

## Model Interface

The preferred sampler interface is an API call that returns structured output:

```json
{
  "diagnosis": "...",
  "edit_plan": ["..."],
  "patch": "...",
  "expected_effect": "...",
  "risk": "..."
}
```

This keeps the orchestrator in control of patch application, rollout, and
logging.

## Training Loop

The initial training loop should be offline:

1. Sample many patch candidates from strong code models.
2. Run rollouts and compute rewards.
3. Build a transition dataset.
4. Train SFT and preference baselines.
5. Compare trained samplers against the original sampler.

Online RFT should only be attempted after reward stability is demonstrated on
cheap tasks.
