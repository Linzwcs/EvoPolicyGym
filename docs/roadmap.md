# HL-Bench Roadmap

## Current State

The current pilot is a single Gymnasium scenario:

```text
MiniGrid-DoorKey-16x16-v0
```

A harder MiniGrid scenario has also been added for the next experiment:

```text
MiniGrid-KeyCorridorS6R3-v0
```

It supports:

- public-observation rollout through the Gymnasium adapter;
- train-only learner workspace artifacts;
- raw public replay logs for each train trial;
- one-step Codex harness runs;
- persistent-workspace multi-epoch Codex loops;
- external train/validation/heldout evaluation;
- accepted/rejected transition logging;
- checkpoint snapshots;
- static HTML/SVG report export.

The most recent 5-epoch run produced a heldout mean-return curve:

```text
0.0000 -> 0.4200 -> 0.8864 -> 0.8990 -> 0.9189 -> 0.9189
```

## Near-Term Engineering

1. Add deterministic train-batch scheduling.

   The train pool should be large, but each epoch should expose only a bounded
   train batch to the learner. This prevents fixed-prefix overfitting and gives
   the learner new train variation across epochs.

2. Add code-complexity metrics.

   Start with dependency-free AST metrics:

   ```text
   nonempty LOC
   function count
   max function length
   AST node count
   branch count
   loop count
   import count
   ```

   Record before/after values in each transition and keep reward primary on
   heldout game return.

3. Harden Codex event capture.

   Current runs capture stdout, stderr, final JSON, patch, and transition. The
   next step is structured event JSONL for command/file/action traces.

4. Separate learner-visible and evaluator-only artifacts more explicitly.

   The workspace should contain train artifacts only. Validation and heldout
   artifacts should remain under evaluator-owned step directories.

5. Add report aggregation across runs.

   A single-run HTML report is implemented. The next report should compare
   multiple seeds, agents, and epoch budgets.

## Benchmark Expansion

1. MiniGrid task suite.

   Add variants that require different heuristic mechanisms:

   ```text
   DoorKey sizes
   KeyCorridor
   Unlock
   UnlockPickup
   KeyCorridor
   FourRooms
   SimpleCrossing
   LavaCrossing
   DynamicObstacles
   Memory
   MultiRoom
   ```

2. Larger split pools.

   A reasonable next scale is:

   ```text
   train pool: 100-300 seeds
   validation: 100 seeds
   heldout: 300-1000 seeds
   ```

   Learner-visible budget remains much smaller than the pool.

3. Multiple run seeds.

   Each benchmark result should average over independent harness runs, not only
   one Codex trajectory.

4. Baselines.

   Keep baselines simple at first:

   ```text
   trivial S0
   random action
   hand-coded heuristic
   one-step Codex
   multi-epoch Codex
   ablations without replay
   ablations without persistent memory
   ```

## Research Questions

1. Does raw replay access improve heldout return AUC?
2. Do stronger agents choose better train experiments under the same budget?
3. Does persistent memory reduce repeated mistakes across epochs?
4. Does complexity control prevent overfitting without suppressing useful
   heuristic growth?
5. Do transition traces train better future heuristic learners?

## Milestones

### M1: Robust Single-Scenario Benchmark

- deterministic train-batch scheduling;
- complexity metrics;
- report export;
- multiple 5-epoch runs on DoorKey-16x16;
- documented acceptance and scoring protocol.

### M2: MiniGrid Suite

- 10-20 MiniGrid variants;
- normalized per-env scores;
- aggregate heldout return AUC;
- per-task reports and suite-level report.

### M3: Training Data Release Candidate

- clean transition JSONL format;
- event trace capture;
- accepted and rejected transition examples;
- metadata for sampler, budget, env version, and artifact hashes.

### M4: Learning From HL Traces

- SFT from high-quality accepted transitions;
- preference pairs from accepted vs rejected candidates;
- offline RL/RFT experiments using heldout-return deltas as reward signals.
