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
from evopolicygym_cartpole import CartPoleBenchmark

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
from evopolicygym_cartpole import CartPoleBenchmark

result = run(
    Program.from_directory("policy/"),
    CartPoleBenchmark(),
    agent=Codex(model="gpt-5.5"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    config=RunConfig(
        split="train",
        max_submissions=20,
        episode_budget=1_000,
        seed=42,
    ),
)
```

The Host owns the Agent task, workspace rules, budget, submit and finish
commands, process supervision, and publication. A provider translates the
Host task into a validated invocation; it does not redefine Run semantics.
The Agent chooses how many of the remaining Episode units to spend on each
submission by default. Set `max_episodes_per_submission` to a positive integer
only when the Host needs an additional cap; its default is `None`.

## Submission accounting

One accepted submission:

1. freezes the current `workspace/program/` tree into a `Program`;
2. reserves and deducts the requested Episode budget;
3. evaluates that immutable snapshot;
4. atomically retains Program, Feedback, Episode summaries, and artifacts;
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
- [Read the CartPole package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/cartpole)
