---
locale: en
page: evaluation
section: api
title: "Evaluation"
navTitle: "Evaluation"
description: "Evaluate one immutable Program over a deterministic Episode plan."
lead: "Use evaluate() when the Program is already fixed."
index: D5
order: 5
docsVersion: v0.3
status: draft
---

## Basic usage

```python
from cartpole import CartPoleBenchmark
from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution

result = evaluate(
    Program.from_directory("my-policy/"),
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
```

`evaluate()` accepts one `Program`, one structural `Benchmark`, an explicit
execution selection, and an optional `EvaluationConfig`.

## EvaluationConfig

| Parameter | Default | Meaning |
| --- | --- | --- |
| `split` | `"validation"` | Benchmark-defined Episode split. |
| `episodes` | `1` | Positive number of Episodes. |
| `seed` | `0` | Unsigned 64-bit seed for Episode planning. |
| `episode_timeout_seconds` | `30.0` | Positive timeout for each Episode. |

The Benchmark must return exactly the requested number of Episodes. Planning
must be deterministic for the same split, seed, and count.

## EvaluationResult

| Field | Meaning |
| --- | --- |
| `benchmark_id` | Public Benchmark identity. |
| `environment_digest` | Identity of the public Environment parameters. |
| `program_digest` | Identity of the evaluated Program. |
| `feedback` | Benchmark-defined score, public content, and artifacts. |
| `episodes` | Sanitized public Episode summaries. |

Episode summaries contain status, total reward, step count, and an optional
Policy failure code. They do not contain scenarios, Environment seeds, Host
paths, credentials, or private metrics.

## Episode behavior

Each Episode receives a fresh Environment, Policy process, Policy instance,
and scratch directory. Policy state may persist only between `act()` calls in
that Episode.

Policy failures produce sanitized failed Episode summaries. Environment,
Benchmark, execution, and cleanup faults abort the Evaluation.

:::warning Local process execution

`ProcessExecution.unsafe()` is not a sandbox. The Policy runs with your
operating-system user permissions.

:::

## Next

- [Programs](./programs.md)
- [Policy API](./policy.md)
- [Coding Agent Runs](./runs.md)
- [Execution and safety](./runtime.md)
