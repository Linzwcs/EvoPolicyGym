# Coding Agent Runs

Configure a bounded caller-owned Run only after the initial Program succeeds
under direct Evaluation.

## Contents

1. Configure the Run
2. Select explicit Agent Skills
3. Preserve the Agent Session boundary
4. Interpret Feedback and Host phases
5. Verify and report

## Configure the Run

Use the public SDK and a new `record_to` path whose parent already exists:

```python
from example_benchmark import ExampleBenchmark, baseline_program

from evopolicygym import (
    AssessmentConfig,
    RunConfig,
    ValidationConfig,
    run,
)
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import ConsoleProgress
from evopolicygym.skills import AgentSkill

result = run(
    baseline_program(),
    ExampleBenchmark(),
    agent=Codex(
        model="MODEL_ID",
        reasoning_effort="high",
    ),
    execution=ProcessExecution.unsafe(),
    record_to="runs/example-001",
    skills=(
        AgentSkill.from_directory("skills/example-policy"),
    ),
    config=RunConfig(
        split="train",
        max_submissions=16,
        episode_budget=48,
        episode_pool_size=96,
        max_episodes_per_submission=4,
        validation=ValidationConfig(
            split="validation",
            episodes_per_candidate=10,
            max_candidates=3,
        ),
        assessment=AssessmentConfig(
            split="test",
            episodes=20,
        ),
        seed=0,
        episode_timeout_seconds=30,
        agent_timeout_seconds=3600,
    ),
    observer=ConsoleProgress(),
)
```

- Never reuse or pre-create the Run directory.
- Set finite submission, Episode, and time budgets.
- Use an explicit per-Submission Episode cap when the selected Benchmark has
  trace-heavy or otherwise batch-sensitive Feedback.
- Choose `episode_pool_size` independently from `episode_budget` when the
  Agent needs more fixed matched conditions than it can spend.
- Add `ConsoleProgress()` only as a non-authoritative observer. Observer
  failure must not change Run semantics.
- State that `ProcessExecution.unsafe()` does not isolate the Agent or Policy.

## Select explicit Agent Skills

Capture each task-specific Skill with `AgentSkill.from_directory()` and pass
it explicitly to `run(..., skills=...)`.

Each Skill is a complete immutable directory snapshot projected read-only at
`workspace/skills/<name>/` and retained by digest. Do not rely on repository
discovery, attach a Skill to `BenchmarkSpec`, or expose Skill content to the
Policy. Do not pass this general caller-side Skill as a replacement for a
Benchmark-specific Policy workflow.

## Preserve the Agent Session boundary

The Host creates the workspace, projects the Agent-only
`evopolicygym-session` client, and generates the current instructions for
submission, candidate handoff, budgets, and paths.

- Treat those Host-generated instructions as authoritative.
- Do not invoke Agent Session commands from the external operator workflow.
- Do not duplicate Session command syntax in provider integrations or
  Benchmark-specific metadata.
- Treat public Episode selectors as indices into one fixed Host-owned training
  pool. Reusing an index consumes budget again while preserving its hidden
  Episode specification and Policy seed.
- Treat a successful candidate handoff as the end of Agent authority. Host
  Validation and Assessment begin only after Agent cleanup.

## Interpret Feedback and Host phases

- Read Agent-visible public Feedback by its documented relative paths.
- Compare Program revisions on matched `episode_index` values. Different index
  sets are unmatched noisy evidence.
- Treat failed trusted evaluation attempts as consumed budget.
- Use `ValidationConfig` to compare ordered candidates on identical private
  Validation Episodes. Never publish that evidence into the Agent workspace.
- Use `AssessmentConfig` only to measure the selected Program on an
  independently seeded held-out split. Assessment never reranks candidates or
  provides a fallback.

## Verify and report

Inspect the retained terminal result and `run.json`. Report:

- selected Benchmark and environment digest;
- initial and final Program digests;
- Run split, seed, submission and Episode budgets, and pool size;
- selected task-specific Skill names and digests;
- ordered candidate IDs;
- Validation configuration and aggregate selection result;
- Assessment configuration and aggregate measurement;
- terminal reason and final submission;
- verification commands and the remaining lack of process isolation.
