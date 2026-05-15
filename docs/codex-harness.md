# Codex Harness Plan

## Purpose

Use Codex itself as the test harness for heuristic-learning experiments.

This is different from using a raw model API as a patch sampler. In harness
mode, Codex is allowed to inspect files, edit `policy.py`, run rollout commands,
read generated summaries, and iterate within a bounded budget.

The outer experiment runner still owns isolation, run directories, seed splits,
final scoring, and data capture.

## Why Use Codex as Harness

Codex harness testing measures the real agentic ability we care about:

- reading a task repository;
- deciding which rollout command to run;
- interpreting failure artifacts;
- editing policy code;
- rerunning tests;
- stopping with a structured report.

This is closer to the learning-beyond-gradients setup than a single API call
that only returns a patch.

## Two Harness Modes

### Mode A: One-Step Harness

Codex performs exactly one improvement step.

```text
H_t
  -> codex exec
  -> edit policy.py
  -> run allowed rollout commands
  -> final structured report
  -> outer runner computes reward
  -> H_{t+1}
```

This is best for training data because each Codex session corresponds to one
transition.

### Mode B: Multi-Step Harness

Codex performs several iterations inside one session.

```text
H_0
  -> codex exec
  -> edit/run/evaluate/edit/run/evaluate
  -> final policy and report
```

This is best for measuring end-to-end Codex HL ability, but it is harder to
assign credit to individual edits.

## Recommended Initial Choice

Start with Mode A.

One-step harness gives cleaner transition data:

```json
{
  "state": "H_t",
  "codex_event_stream": "events.jsonl",
  "diff": "policy.patch",
  "rollout_summary": "summary.json",
  "reward": "reward.json",
  "next_state": "H_t_plus_1"
}
```

## Current Pilot Entry Point

The implemented Minigrid pilot runner is:

```text
python -m hlbench.harness.run_codex_step --scenario minigrid_doorkey --run-id <run_id>
```

It performs the complete one-step loop:

```text
create workspace
run before train/validation/heldout evaluator
copy train feedback into learner workspace
run codex exec as a subprocess
compile final policy
run after train/validation/heldout evaluator
write policy.patch and transition.json
```

Use this mode to test the evaluator and transition logger without invoking
Codex:

```text
python -m hlbench.harness.run_codex_step --scenario minigrid_doorkey --run-id <run_id> --skip-codex
```

The current implementation captures Codex stdout and stderr under
`runs/<run_id>/codex/`. JSON event streaming and schema-constrained CLI output
remain future hardening work.

For repeated optimization, use `epochs` as an outer-loop hyperparameter:

```text
python -m hlbench.harness.run_codex_loop --scenario minigrid_doorkey --run-id <run_id> --epochs 5
```

This keeps a single persistent learner workspace at `runs/<run_id>/workspace/`.
The benchmark owns versioning by copying workspace snapshots into
`runs/<run_id>/checkpoints/H_*/workspace/` and writing per-epoch transition
records under `runs/<run_id>/steps/epoch_*/`.

The multi-epoch loop is continuous: every epoch's candidate workspace becomes
the next epoch's starting point and is checkpointed as the next `H_*` version.
`accepted=false` remains useful as an evaluation label, but it does not trigger
rollback. This preserves failed or partial attempts so the learner can repair
or build on them in later epochs.

The prompt should describe available raw artifacts and access boundaries, not
prescribe replay analysis, regression-test selection, state detectors, or memory
maintenance strategies. Those behaviors should emerge from the learner's own
use of the workspace.

## Codex Exec Interface

Codex CLI supports non-interactive execution through `codex exec`.

Use:

```text
codex exec
  --cd <candidate_workspace>
  --sandbox workspace-write
  --ask-for-approval never
  --json
  --output-schema schemas/codex-harness-result.schema.json
  -o runs/<run_id>/codex-final.json
```

`--json` streams Codex events as JSONL. The event stream is useful for training
and auditing because it records commands, messages, file changes, and terminal
results.

`--output-schema` constrains the final Codex message so the outer runner can
parse it deterministically.

## Workspace Contract

Each Codex harness run should receive a temporary workspace:

```text
candidate_workspace/
  AGENTS.md
  task_spec.md
  policy.py
  tools/
    run_rollout.py
    summarize_rollout.py
    compute_reward.py
  data/
    train_instances.jsonl
    validation_instances.jsonl
  runs/
```

Allowed edits:

```text
policy.py
policy_memory.json
notes.md
```

Disallowed edits:

```text
tools/
data/
schemas/
task_spec.md
```

The outer runner should verify this with git diff or file hashes after Codex
finishes.

## Prompt Contract

The prompt should be narrow and operational:

```text
You are running one heuristic-learning improvement step.

Allowed goal:
  Improve policy.py based on train rollout feedback.

Allowed commands:
  python tools/run_rollout.py --split train --budget cheap
  python tools/summarize_rollout.py <run_dir>

Required behavior:
  - Edit only policy.py, policy_memory.json, or notes.md.
  - Do not inspect validation or held-out data.
  - Do not modify tools, data, schemas, or task_spec.md.
  - Run at least one train rollout after editing.
  - Stop after one accepted policy edit.
  - Return final JSON matching the schema.
```

## Outer Runner Responsibilities

The outer runner launches Codex and then independently evaluates the result.

Responsibilities:

1. Create isolated workspace.
2. Copy task files and current policy.
3. Run public train feedback and copy only train artifacts into the workspace.
4. Record pre-run hashes of protected files.
5. Run `codex exec`.
6. Save `events.jsonl` and `codex-final.json`.
7. Verify protected files were not changed.
8. Extract final policy diff.
9. Re-run the saved before policy and the candidate policy on private
   validation and held-out splits outside Codex.
10. Compute reward.
11. Append transition record.

Codex may run train rollouts, but final reward should be computed by the outer
runner. Validation and held-out rollouts should not write replay files by
default; train replays are public feedback.

## Epoch Logging

Every automated run should keep both raw Codex output and structured epoch
events.

Run-level files:

```text
runs/<run_id>/events.jsonl
runs/<run_id>/transitions.jsonl
runs/<run_id>/learning_curve.json
```

Per-epoch files:

```text
runs/<run_id>/steps/epoch_000/epoch_events.jsonl
runs/<run_id>/steps/epoch_000/epoch_summary.json
runs/<run_id>/steps/epoch_000/transition.json
runs/<run_id>/steps/epoch_000/policy.patch
runs/<run_id>/steps/epoch_000/codex/stdout.txt
runs/<run_id>/steps/epoch_000/codex/stderr.txt
runs/<run_id>/steps/epoch_000/codex/final.json
runs/<run_id>/steps/epoch_000/codex/run.json
```

`epoch_events.jsonl` is the stage timeline for one epoch: train feedback,
prompt creation, Codex start/end, compile result, private evaluator runs,
reward computation, accept/reject label, continuous checkpoint update, and
transition write.

`codex/stdout.txt` and `codex/stderr.txt` store the raw Codex process output.
`codex/run.json` records command metadata, elapsed time, return code, timeout
state, output byte counts, and paths to the raw output files.

These logs are evaluator-side artifacts. The learner workspace receives only
public train feedback under `workspace/rollout/` and learner-created train
history under `workspace/rollouts/`.

## Example Driver Shape

```python
def run_codex_harness(run_id: str, workspace: Path, prompt_path: Path) -> HarnessResult:
    events_path = Path("runs") / run_id / "codex-events.jsonl"
    final_path = Path("runs") / run_id / "codex-final.json"

    cmd = [
        "codex",
        "exec",
        "--cd",
        str(workspace),
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "--json",
        "--output-schema",
        "schemas/codex-harness-result.schema.json",
        "-o",
        str(final_path),
        "-",
    ]

    prompt = prompt_path.read_text()
    completed = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1800,
        check=False,
    )
    events_path.write_text(completed.stdout)
    return parse_harness_result(final_path, completed)
```

## Training Data From Codex Harness

Codex harness generates richer trajectories than a patch-only sampler.

Useful records:

- final diff;
- command execution sequence;
- rollout summaries Codex inspected;
- failed commands;
- file changes;
- final report;
- outer validation reward;
- held-out reward;
- protected-file violation flag.

This supports two training targets:

1. Patch generator training from accepted final diffs.
2. Agent/harness training from full command-and-edit trajectories.

## Evaluation Metrics

Measure Codex harness as an agent:

- held-out score improvement per harness run;
- number of train rollouts used;
- invalid edit rate;
- protected-file violation rate;
- time to first valid improvement;
- regression rate;
- final code complexity;
- event trace length and command count.

## Main Tradeoff

Codex harness mode is more realistic but less clean.

Patch-only API sampling gives cleaner supervised/RFT examples. Codex harness
gives more faithful measurements of iterative agent behavior. The experiment
should support both, but use Codex harness as the primary evaluation of HL
ability.
