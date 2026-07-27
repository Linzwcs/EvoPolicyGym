---
locale: en
page: evaluation
section: core
title: "Evaluation and Runs"
navTitle: "Evaluation and Runs"
description: "Direct Program evaluation and bounded Coding Agent Program-evolution Runs."
lead: "Use evaluate() for one immutable Program, or run() to let a Coding Agent author and submit multiple candidates."
index: D4
order: 4
docsVersion: v0.3
status: draft
---

## Direct Evaluation

`evaluate()` evaluates one immutable `Program` against one structural
`Benchmark`:

```python
from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution
from cartpole import CartPoleBenchmark

result = evaluate(
    Program.from_directory("policy/"),
    CartPoleBenchmark(),
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(
        split="validation",
        episodes=100,
        seed=42,
        episode_timeout_seconds=30,
    ),
)

print(result.feedback.score)
print(result.episodes)
```

`EvaluationConfig` is immutable and finite. The Benchmark must plan exactly the
requested number of deterministic Episodes.

## EvaluationResult

An `EvaluationResult` contains:

| Field | Meaning |
| --- | --- |
| `benchmark_id` | Stable public Benchmark identity. |
| `environment_digest` | Canonical identity of the applied public Environment parameters. |
| `program_digest` | SHA-256 identity of the evaluated Program. |
| `feedback` | Benchmark-defined score, public content, and optional artifacts. |
| `episodes` | Sanitized public Episode summaries. |

Episode summaries never expose trusted scenario values, Environment seeds,
Host paths, credentials, or private runtime evidence.

## Program-Evolution Run

`run()` gives one `CodingAgent` bounded authority to improve an initial
Program:

```python
from evopolicygym import Program, RunConfig, run
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from cartpole import CartPoleBenchmark

result = run(
    Program.from_directory("policy/"),
    CartPoleBenchmark(),
    agent=Codex(
        model="gpt-5.5",
        reasoning_effort="high",
    ),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    config=RunConfig(
        split="train",
        max_submissions=20,
        episode_budget=1_000,
        episode_pool_size=2_000,
        seed=42,
    ),
)
```

The Host owns the Agent task, workspace rules, budget, submit and finish
commands, process supervision, and publication. A provider translates the
Host task into a validated invocation; it does not redefine Run semantics.
Before Agent execution, the Host deterministically builds one fixed training
Episode pool from `RunConfig.seed`. The Agent submits public Run-local
singleton and half-open range unions such as:

```console
evopolicygym submit program --episodes "0:2,4:8"
```

That selector expands to indices `0, 1, 4, 5, 6, 7`. The same index preserves
its hidden Episode specification and Policy seed across Submissions, but every
use creates a fresh Environment and Policy runtime and consumes budget again.
`episode_pool_size` defaults to `episode_budget`. Set
`max_episodes_per_submission` only when the Host needs an additional cap.

Agent Skills are independent Run inputs rather than Benchmark properties.
Freeze each selected directory with
`AgentSkill.from_directory("skills/<name>")` and pass the snapshots through
`run(..., skills=(skill,))`. The Host exposes complete Skill directories
read-only under `workspace/skills/`, records their names and content digests in
`run.json`, and never passes them into a Policy process. The repository's
Balatro Skill is one optional experimental condition for evidence allocation,
strategy development, and Policy hardening.

## Submission accounting

One accepted submission:

1. freezes the current `workspace/program/` tree into a `Program`;
2. validates the selected training indices, then reserves and deducts one
   Episode budget unit per index;
3. evaluates that immutable snapshot;
4. atomically retains Program, selected indices, Feedback, Episode summaries,
   and artifacts;
5. publishes an independent Agent-visible copy under `workspace/feedback/`;
6. admits the Submission ID as a possible final selection.

Invalid Program capture does not consume Episode budget. Once Evaluation
starts, reserved budget is not refunded. A Policy failure is a committed scored
result; a trusted Evaluation fault closes the Run as `evaluation_failed`.

## Selecting the final Program

The Agent finishes by selecting one fully published Submission. The returned
`RunResult.final_program` is the detached Program retained for that
Submission—not the possibly modified contents left in the Agent workspace.

Possible terminal reasons are:

- `finished`
- `agent_exited`
- `budget_exhausted`
- `agent_failed`
- `evaluation_failed`

## Run records

A local Run retains Programs, Feedback, artifacts, events, Agent invocation and
logs, and a terminal `run.json` manifest. The record is diagnostic and
reproducible within the current design, but it is not resumable: v0.3 has no
durable ledger, crash recovery, or resume protocol.

[Inspect the Run record layout →](../../runs/)

## Next

- [Execution and safety →](../runtime/)
- [Benchmark authoring →](../authoring/)
- [Read the CartPole package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/cartpole)
- [Read the Acrobot package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/acrobot)
- [Read the Mountain Car package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/mountain_car)
- [Read the Continuous Mountain Car package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/mountain_car_continuous)
- [Read the Pendulum package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/pendulum)
- [Read the Balatro package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/jackdaw/balatro)
