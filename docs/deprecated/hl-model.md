# Heuristic Learning Model

## Source Framing

This document models Heuristic Learning (HL) in the sense of Jiayi Weng's
*Learning Beyond Gradients*: a coding agent repeatedly maintains a programmatic
heuristic system using feedback from rollouts, tests, logs, replays, videos, and
human notes.

The key distinction is:

```text
Deep RL:
  update neural-network parameters through gradient-based training

HL:
  update a software system through agent-authored edits
```

The learned object is not just a `policy.py`. It is a Heuristic System (HS):
code, state representation, memory, feedback channels, experiments, tests,
regressions, and an update path that lets the next coding agent continue.

## Two-Time-Scale Model

HL has an inner execution loop and an outer maintenance loop.

### Inner Loop: Heuristic Execution

At environment time `k`, the current HS executes a policy:

```text
a_k = pi_{S_t}(o_k, h_k)
```

where:

- `S_t` is the current Heuristic System version;
- `o_k` is the environment observation;
- `h_k` is allowed runtime history, caches, detector state, or local memory;
- `a_k` is the task action.

The inner loop produces trajectories, scores, logs, failures, traces, and
replays.

### Outer Loop: Heuristic Learning

At update time `t`, a coding agent observes a bounded context and edits the HS:

```text
C_t = Obs(S_t, F_t, G, K_t)
A_t ~ pi_theta(. | C_t)
S_{t+1} = Apply(S_t, A_t)
Y_t = Eval(S_{t+1})
R_t = Reward(S_t, A_t, Y_t)
```

where:

- `F_t` is feedback from rollouts, tests, logs, replays, videos, and humans;
- `G` is the task goal and evaluation contract;
- `K_t` is accumulated history such as accepted patches, rejected directions,
  regressions, and known failure modes;
- `A_t` is an agent-authored update;
- `Y_t` is the independent evaluation result;
- `R_t` is the training or selection reward for the update.

The sampler model `pi_theta` is trained to become better at producing useful HS
updates under fixed budgets and safety constraints.

## Heuristic System State

Model a Heuristic System as:

```text
S_t = (P_t, Z_t, M_t, Q_t, L_t, U_t)
```

where:

- `P_t`: executable policy, controllers, state machines, planners, macro-actions,
  and helper code;
- `Z_t`: readable state representation, detectors, feature extractors, caches,
  and probes;
- `M_t`: explicit memory, notes, summaries, rejected hypotheses, and discovered
  invariants;
- `Q_t`: regression tests, fixed-seed replays, golden traces, failure cases, and
  validation checks;
- `L_t`: experiment logs, rollout summaries, videos, metrics, and version diffs;
- `U_t`: update protocol, allowed files, sandbox rules, prompt templates, and
  orchestration scripts.

This definition is intentionally broader than a task policy. HL is only
well-defined when feedback and future update paths are part of the maintained
system.

## Feedback

Feedback is any bounded artifact that can improve the next update:

```text
F_t = (score_t, failures_t, traces_t, tests_t, logs_t, replays_t, human_t)
```

Feedback should answer:

- what changed;
- what improved;
- what regressed;
- why the current HS failed;
- which failures are reproducible;
- which old behaviors must be preserved.

For train-time context, feedback may include representative train failures. For
final reporting, held-out scores may be used. Held-out failure details should
not be shown to the sampler.

## Update Actions

A full HL action can edit several HS components:

```text
A_t = (
  diagnosis,
  experiment_plan,
  code_patch,
  test_or_replay_patch,
  memory_update,
  simplification_or_refactor,
  expected_effect,
  risk
)
```

For early controlled experiments, restrict actions to a smaller surface:

```text
A_t = (diagnosis, edit_plan, unified_diff, expected_effect, risk)
```

This restricted action maps to
[`schemas/patch-proposal.schema.json`](../schemas/patch-proposal.schema.json).
The broader action space is the long-term target once the runner can safely
separate policy edits, test additions, memory updates, and evaluator internals.

## Transition Semantics

The outer transition is agent-mediated software evolution:

```text
S_t --A_t--> S'_t --evaluation--> (S_{t+1}, Y_t)
```

The transition procedure is:

1. apply the proposed update in an isolated workspace;
2. reject edits outside the allowed HS surface;
3. run static checks and regression tests;
4. run bounded train and validation rollouts;
5. summarize failures and metrics;
6. compute reward independently from sampler rationale;
7. accept, reject, or rank the candidate;
8. serialize the transition for training.

This maps to `state`, `action`, `result`, and `reward` in
[`schemas/transition.schema.json`](../schemas/transition.schema.json).

## Objective

The target is not one good final policy. The target is a sampler that improves
HS versions faster and more safely:

```text
maximize_theta E[
  AUC(heldout_score over update budget)
  - cumulative_rollout_cost
  - cumulative_regressions
  - invalid_update_cost
  - coupling_complexity_growth
]
```

This objective prefers update processes that:

- improve under the same rollout budget;
- preserve old capabilities through tests and replays;
- turn failures into targeted code or memory changes;
- avoid seed-specific hacks and evaluator leakage;
- compress accumulated patches into simpler system structure.

## Absorb And Compress

A healthy HS needs two recurring operations.

### Absorb Feedback

Absorption converts new experience into system material:

```text
new failure -> detector, guard rule, test, replay, note, or metric
```

Absorption increases capability but often increases coupling complexity.

### Compress History

Compression turns local fixes into simpler reusable structure:

```text
many local patches -> cleaner state representation, shared primitive, simpler
controller, regression suite, or deleted obsolete branch
```

Compression is required because an HS that only grows eventually becomes too
coupled for the agent to maintain.

The modeling implication is that reward cannot only measure score gain. It must
also price maintainability, regression coverage, and future update capacity.

## Coupling Complexity

Let `Kappa(S_t, pi_theta, tools)` be the effective coupling complexity that the
current agent-tool stack must manage.

`Kappa` increases with:

- interdependent rules;
- hidden state;
- unstable interfaces;
- weak observability;
- missing regression tests;
- long feedback latency;
- undocumented patch history.

`Kappa` decreases with:

- modular detectors and controllers;
- stable policy interfaces;
- reproducible seeds and replays;
- concise summaries;
- focused tests;
- searchable logs;
- explicit memory of failed directions.

A practical HL system should track proxy metrics rather than pretend there is a
single exact complexity scalar:

- static complexity and branch count;
- module boundary violations;
- regression-test coverage;
- number of active failure modes;
- average localization time;
- rate of patches that fix one case and break another;
- summary size required for the next successful update.

## Training Views

The same transition records support several learning objectives.

### SFT

```text
input:  C_t
target: A_t
```

Use high-quality updates that improve validation behavior, avoid violations, and
do not create unnecessary coupling.

### Preference Training

```text
(C_t, A_i, A_j, label)
```

Prefer updates with better validation reward, fewer regressions, clearer
failure-to-patch alignment, and better maintainability.

### Offline RL

```text
(C_t, A_t, R_t, C_{t+1}, done)
```

This becomes useful after reward noise and invalid update handling are stable.

### Online RFT

```text
sample update -> apply in sandbox -> run bounded evaluation -> scalar reward
```

Only cheap task families should be used until sandboxing and reward stability
are proven.

## MVP Instantiation

For this repository's first experiments:

```text
S_t:
  policy.py
  optional policy_memory.json
  accepted and rejected patch history
  train failure summaries
  rollout summaries
  fixed task contract

C_t:
  task spec excerpt
  allowed files
  current policy code
  rollout/summary.json
  selected train failures
  previous patch outcomes
  constraints and budget

A_t:
  one unified diff over allowed files

Y_t:
  patch application result
  static check result
  train and validation rollout summaries
  held-out reporting summary
  reward components
```

Start with a restricted HS surface because it gives clean attribution. Once the
transition factory is reproducible, expand the action space to include tests,
replays, memory, and simplification patches.

## Core Research Questions

1. Can training improve the sampler's ability to maintain an HS under the same
   update and rollout budget?
2. Does explicit feedback memory reduce repeated failed directions?
3. Do regression tests and replays reduce catastrophic forgetting relative to
   score-only patching?
4. Does compression improve long-horizon learning curves, even when it gives
   little immediate score gain?
5. Which task families expose the limit of code-based heuristics and require
   neural components?

## Reference

- Jiayi Weng, *Learning Beyond Gradients*,
  <https://trinkle23897.github.io/learning-beyond-gradients/>.
