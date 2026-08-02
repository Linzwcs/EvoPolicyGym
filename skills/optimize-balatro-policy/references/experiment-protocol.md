# Train-only experiment protocol

In a formal EvoPolicyGym Run, the Agent receives only train submissions from
one fixed Host-owned Episode pool. The Host exposes Run-local integer indices,
not the hidden Environment or Policy seeds behind them. The Agent may select
and reuse indices across submissions; the same index preserves its hidden
Episode specification and Policy seed while every use creates fresh runtime
state and consumes budget again. Host Validation and Assessment start only
after `finish` closes Agent authority, use independent Host-only Episodes, and
never return evidence to the workspace.

Create evidence roles by partitioning the train budget, not by pretending to
have Benchmark validation data.

## Contents

- Partition before the first submission
- Run development experiments
- Enter frozen confirmation once
- Report metrics and hand off to the Host
- Keep an evidence ledger

## Partition before the first submission

Read the available Run-local index range, total Episode budget,
per-submission cap, maximum submissions, and candidate limit. Pool size is the
number of selectable conditions; budget is the total number of evaluations,
including repeated indices. Write down both an index allocation and a spend
allocation before seeing results. A useful starting point is:

| Role | Approximate budget | Index policy | Replay use |
|---|---:|---|---|
| Baseline and correctness | 10% | small declared baseline set | inspect |
| Diagnosis and architecture | 20% | declared development sets | inspect |
| Capability experiments | 45% | matched control/candidate selectors | inspect |
| Frozen confirmation | 25% | pre-reserved unseen train indices | aggregates only |

Adapt percentages to the budget, but reserve at least two meaningful
submissions for frozen confirmation when limits allow, or one when the budget
is very small. Reserve their indices before development and do not inspect
them early. Never borrow the entire reserve to continue an attractive
experiment. Record every submission's preassigned role and exact selector in
the evidence ledger. Keep that ledger in working reasoning rather than adding
non-Policy artifacts to the submitted `program/`.

## Run development experiments

1. Record the control Program digest and an exact inverse edit or restore
   method allowed by the workspace. Do not add backups inside `program/`.
2. State one hypothesis, eligible public states, decisions it may change,
   safety guards, and rejection conditions.
3. Before spending Episodes, use development replay or a counterfactual audit.
   Count eligible states and changed actions. Drop behavior-neutral ideas unless
   they are correctness or architecture work.
4. Choose a small, predeclared development selector. Submit the exact control
   digest and candidate digest on the same indices. Restore and verify the
   control digest before its submission when needed.
5. Compare outcomes only after matching each public `episode_index`. This is a
   matched train A/B, not access to the hidden seed or Case.
6. If promising, repeat the comparison on a second predeclared selector. Pool
   only identical digests, retain per-index deltas, and check consistency
   across selectors.
7. Treat results from different selectors as unmatched noisy evidence. Never
   claim a paired improvement unless the index sets match exactly.
8. Revert rejected work exactly and verify the restored digest.

Use replay counterfactuals for action-level comparison and repeated immutable
digests for environment-level confidence.

## Enter frozen confirmation once

Before consuming the reserved confirmation budget:

1. Stop adding capabilities.
2. Freeze a credible shortlist of immutable digests represented by published
   submissions and the rule for ordering them.
3. Evaluate each feasible frozen candidate on the same pre-reserved,
   previously unseen train selector. Verify every restored digest before
   measuring it.
4. Inspect only terminal aggregates. Do not open confirmation replay to design
   another patch.
5. Use confirmation to reject brittle candidates or adjust shortlist order,
   not to resume threshold tuning. At `finish`, include at most one published
   submission ID for each distinct candidate digest.

Confirmation is still train evidence, not true validation. Its protection comes
from reserving it before development and stopping adaptive edits when it begins.
If a Policy failure appears, prefer a previously published safe candidate.

## Report the right metrics

Balatro run reward is `rounds_cleared + 1000` on a win. A completed Ante 8 run
therefore normally has reward 1024. Report reward and progress separately:

- immutable Program digest, submission IDs, evidence role, selected indices,
  and Episodes;
- wins and Policy failures;
- mean and median Blinds cleared;
- early deaths (`≤5`);
- mid-game completions (`≥12`);
- late-game completions (`≥18`);
- repeated-submission consistency for the same digest.

A rare win demonstrates ceiling, not a stable win rate. Do not rank candidates
only by the largest submission mean.

## Hand off to the Host

Consume the legal Episode budget, then call `finish` with the strongest
published submission or ordered credible shortlist allowed by the task. The
Host evaluates shortlisted Programs on identical private Validation Episodes
after Agent exit and selects by the declared metric and tie-breaks. Assessment
then measures only the selected Program and never changes selection.

Do not attempt to reproduce either stage from train Feedback. Never expect,
inspect, or adapt to Host Validation or Assessment evidence.

## Keep an evidence ledger

Use one row per immutable digest, selector, and evidence role:

| Digest | Submission IDs | Selector | Role | Episodes | Mean / median | ≤5 | ≥12 | ≥18 | Wins | Failures | Decision |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|

Link public artifacts and record whether replay was inspected. Keep rejected
experiments so later agents do not repeat the same adaptive overfit.
