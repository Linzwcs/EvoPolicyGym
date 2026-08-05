---
locale: en
page: runs
section: api
title: "Coding Agent Runs"
navTitle: "Runs"
description: "Let one Coding Agent edit, evaluate, and hand off Program candidates under fixed limits."
lead: "A Run gives one Coding Agent a workspace, an Episode budget, and a fixed training pool."
index: D6
order: 6
docsVersion: v0.3
status: draft
---

## Start a Run

```python
from cartpole import CartPoleBenchmark
from evopolicygym import Program
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import RunConfig, run

result = run(
    Program.from_directory("my-policy/"),
    CartPoleBenchmark(),
    agent=Codex(model="gpt-5.6-luna", reasoning_effort="high"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    config=RunConfig(
        max_submissions=3,
        episode_budget=30,
        episode_pool_size=60,
        max_episodes_per_submission=10,
        seed=42,
    ),
)
```

The same Run can select either of the other first-party command-line
integrations without changing the Benchmark or Run configuration:

```python
from evopolicygym.agents import ClaudeCode, KimiCode

claude = ClaudeCode(model="sonnet", effort="high")
kimi = KimiCode(model="kimi-code/kimi-for-coding")
```

All three providers run non-interactively and retain their structured stdout
as Agent evidence. Claude Code runs in bare mode without session persistence.
Kimi Code currently persists CLI state, so set `KIMI_CODE_HOME` to a dedicated
caller-owned directory when Runs must not share its configuration or history.

The Host creates the Episode pool before the Agent starts. The Agent can edit
`workspace/program/`, submit candidates, read committed Feedback, and finish
with published candidates.

## Run limits

| Parameter | Default | Meaning |
| --- | --- | --- |
| `split` | `"train"` | Benchmark split available to the Agent. |
| `max_submissions` | `20` | Maximum accepted Submissions. |
| `episode_budget` | `1_000` | Total Episode indices charged across all Submissions. |
| `episode_pool_size` | Episode budget | Number of fixed Run-local Episode identities. |
| `max_episodes_per_submission` | `None` | Optional cap for one Submission. |
| `seed` | `0` | Seed used to build the training pool. |
| `episode_timeout_seconds` | `30.0` | Timeout for each Episode. |
| `agent_timeout_seconds` | `3_600.0` | Timeout for the Agent process. |

Pool size and budget are separate. Reusing an Episode index supports a matched
Program comparison, but it consumes budget again and still creates fresh
Environment and Policy state.

## Submit and finish

Inside the active Session, the Agent uses:

```console
evopolicygym-session submit program --episodes "0:2,4:8"
evopolicygym-session finish submission-000002
```

The selector expands to indices `0, 1, 4, 5, 6, 7`. Selectors must be non-empty,
strictly increasing, in range, and within the remaining budget.

A completed Submission atomically publishes its Program, selected indices,
Feedback, Episode summaries, and artifacts. Failed Program capture does not
consume Episode budget. Once Evaluation starts, reserved budget is not
refunded.

## Validation and Assessment

Without `ValidationConfig`, the Agent finishes with one published candidate.
With Validation, it may hand off an ordered candidate list. After Agent
cleanup, the Host evaluates every candidate on the same private Validation
Episodes and selects one Program.

`AssessmentConfig` evaluates only that selected Program on a held-out split.
Assessment never changes selection. Validation and Assessment evidence is not
published to the Agent workspace.

## Agent Skills

Agent Skills are explicit Run inputs:

```python
from evopolicygym.skills import AgentSkill

skill = AgentSkill.from_directory("skills/evopolicygym")
result = run(..., skills=(skill,))
```

Skills are copied read-only into `workspace/skills/`. They are never passed to
the Policy process.

## Result and records

`RunResult` contains the terminal reason, published Submissions, handed-off
candidate IDs, selected Program, and optional Validation and Assessment
results.

The directory passed to `record_to` contains the workspace, immutable
Programs, Feedback, artifacts, events, Agent logs, and `run.json`. Runs are not
resumable in `v0.3`.

:::warning Local process execution

`ProcessExecution.unsafe()` is not a sandbox. Agent and Policy code run with
your operating-system user permissions.

:::

## Next

- [Evaluation](./evaluation.md)
- [Execution and safety](./runtime.md)
- [Run record layout](/runs/)
