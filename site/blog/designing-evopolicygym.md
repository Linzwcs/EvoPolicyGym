---
locale: en
page: designing-evopolicygym
title: "EvoPolicyGym: environments for agents that build strategy systems"
description: "Why EvoPolicyGym uses interactive Environments and executable Programs to study and train coding agents."
lead: "A coding agent studies an Environment, learns from bounded Feedback, and turns its conclusions into an executable strategy system."
publishedAt: "2026-07-27"
date: "2026-07-27"
authors: [evopolicygym]
tags:
  - Design
  - Motivation
  - Architecture
status: published
---

## Coding agents as policy-system experts

EvoPolicyGym was directly inspired by Jiayi Weng's
[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/).
The article describes *Heuristic Learning*: a coding agent absorbs rewards,
failures, tests, logs, and replays, then improves a programmatic policy by
editing the software system itself. It showed us that agentic coding can serve
as a learning process whose evolving state is explicit in code.

EvoPolicyGym begins from that insight and asks how to make the process bounded,
reproducible, and comparable across interactive Environments.

<!-- truncate -->

An EvoPolicyGym Run places a coding agent in the role of a policy-system
expert. The agent studies the Environment and its interface, inspects an
initial Program, forms hypotheses about successful behavior, and writes those
ideas into a complete executable Policy system.

That system may combine domain knowledge, state estimation, rules, planning,
search, memory, algorithms, or tuned parameters. The coding agent is free to
change its internal design as evidence accumulates. Its responsibility is to
turn what it learns into source code that can make decisions on its own.

Authoring and execution occupy two distinct phases. During Evaluation, the
submitted Policy independently receives observations and produces Actions.
The result is a separable strategy artifact that can be frozen, inspected,
rerun, and compared.

Environment Feedback closes the engineering loop:

```text
study the Environment
    ↓
author an executable Policy system
    ↓
submit and evaluate it
    ↓
inspect scores, traces, and artifacts
    ↓
diagnose, redesign, and submit again
```

This is the motivation for Autonomous Policy Evolution in EvoPolicyGym. The
coding agent contributes expertise and software engineering; the Environment
contributes empirical evidence; the evolving Program records the resulting
strategy. The central question is how effectively an agent can transform
limited Environment Feedback into a better executable decision system.

## Separate the expert from the Policy

EvoPolicyGym gives the coding agent and the Policy different roles.

The coding agent is the outer policy engineer and optimizer. It reads
instructions and public Feedback, edits `workspace/program/`, and decides when
to submit another candidate. The Policy is the inner decision system. It
receives observations through a small ABI and returns Actions while an Episode
is running.

A Policy may retain state between `act()` calls inside one Episode. Each new
Episode receives a fresh process and Policy instance, while cross-Episode
improvement is represented by a new Program.

EvoPolicyGym locates learning at the Program level between Episodes.
Episode-local state supports temporal behavior; Program revision captures
lasting improvement. Each change therefore has a visible source snapshot and
a clear relationship to its Evaluation evidence.

## Make the artifact first-class

The workspace supports live authoring. Every accepted submission turns its
current source tree into an immutable, content-addressed `Program`. The
Evaluation, Feedback, and artifacts belong to that exact snapshot, and the
final result returns the retained Program selected from submitted candidates.

This choice makes a Run understandable as a sequence of authored artifacts:

| Object | Responsibility |
| --- | --- |
| `Program` | The executable Policy source being evaluated |
| `Submission` | One immutable Program, an explicit training-index selector, and committed Feedback |
| `Run` | A bounded sequence of submissions and a final handoff |
| `Validation` | Host-side selection among finished candidates |
| `Assessment` | Held-out measurement of the selected Program |

The Program is the durable result. The agent transcript and process logs
provide supporting diagnostics.

## Let each Benchmark define useful Feedback

Different Environments expose different kinds of evidence. A control task may
benefit from state trajectories and termination causes. A card game may need
round summaries, economy decisions, or compact replays.

EvoPolicyGym standardizes the Feedback carrier while each Benchmark defines
its useful domain content. Feedback always has a scalar score, and the
Benchmark may add bounded public values and artifacts. The Benchmark also owns
Episode planning, Environment construction, Action validation, and scoring.

The Kernel owns what must remain consistent across Benchmarks: budgets,
immutable submissions, lifecycle ordering, publication, selection, records,
and the Policy ABI. Environment packages remain independently installable and
depend only on the public authoring interface.

This division lets the project grow like a Gym-style ecosystem while the
Kernel remains stable and domain-independent.

## Treat evidence access as part of the experiment

A Run's submission limit, total Episode budget, fixed training-pool size, and
optional per-Submission cap define its experimental condition. For example,
sixteen submissions, forty-eight Episode units, and a pool of ninety-six
Episode identities grant forty-eight total observations selected from a wider
set; the larger pool does not increase the interaction budget.

The Host constructs this indexed pool before the agent starts. Each Submission
names a non-empty set of public Run-local indices. Reusing an index keeps its
hidden Episode specification and Policy seed fixed, so two immutable Programs
can be compared on matched evidence. Every use still creates a fresh
Environment and Policy runtime and consumes budget again. Actual seeds,
scenarios, and pool construction remain Host-owned.

Once an Evaluation begins, its reserved Episode allocation is consumed. Policy
failures and invalid Actions are reported as observed behavior, preserving the
exact semantics of the submitted Program. Feedback maps every sanitized
Episode outcome back to its public index, while comparisons over different
selectors remain unmatched evidence.

The agent uses public search Feedback to decide what to try next. When it
finishes, authority returns to the Host. Private Validation selects among the
handed-off candidates, and held-out Assessment measures the selected Program.
Optimization Feedback closes at `finish`; selection and final measurement
remain Host-side.

Keeping search, selection, and final measurement distinct makes the reported
result easier to interpret.

## Keep the Kernel focused

EvoPolicyGym is infrastructure with deliberately focused ownership.

- Agent integrations translate a Host-owned task into a provider invocation.
- Benchmark distributions own domain semantics, dependencies, baselines,
  Feedback, and tests.
- The Kernel owns the shared evaluation and Program-evolution lifecycle.
- The Policy boundary carries bounded public values.

Today Codex is the first supported coding-agent integration, and local process
execution is the active backend. The contracts remain provider- and
backend-independent, preserving the meaning of Program, Submission,
Evaluation, and Run as more integrations arrive.

## What EvoPolicyGym enables

EvoPolicyGym connects scalable interactive Environments, coding agents,
versioned Programs, and verifiable Benchmark evidence. The Agent is the subject
of study and training; the Program is the executable evidence it leaves
behind.

### Study agents that evolve strategy systems

With the Environment, initial Program, interaction budget, Feedback visibility,
and selection rules held constant, repeated Runs can study:

- **Agent capability:** which coding agent authors the strongest Policy system
  under the same conditions?
- **Improvement efficiency:** how much Environment interaction produces stable
  Program improvement?
- **Feedback value:** which traces, diagnostics, replays, and aggregate signals
  lead to effective revisions?
- **Evolution dynamics:** how does Program structure change across Submissions,
  and which changes produce durable gains?
- **Selection validity:** does the candidate chosen by Validation retain its
  advantage in held-out Assessment?
- **Policy-system design:** which state representations, rules, planners,
  memories, and controllers do different agents encode?
- **Scaling and generalization:** how do these results change across budgets,
  profiles, seeds, task complexity, and Environment families?

The final score measures the Agent's selected artifact. The sequence of
immutable Programs, Feedback, artifacts, and outcomes explains how the Agent
reached it.

### Train coding agents with interactive Environments

An Environment and Benchmark together form a task generator, evidence
generator, and verifier. Profiles, scenarios, and seeds create task variation;
Program evaluations, public Feedback, diagnostics, and held-out outcomes
provide training signals.

The same Environment ecosystem can support:

- coding-agent RL and RLVR using Program evaluation and held-out performance as
  verifiable outcomes;
- SFT from successful long-horizon Agent trajectories;
- agent distillation from observable evolution records: task context, public
  Feedback, Program changes, Submissions, and outcomes;
- rejection sampling of high-quality Agent trajectories based on their final
  artifacts and results;
- curriculum learning across task profiles and difficulty;
- process supervision from intermediate failures, revisions, and evaluations.

An Agent evolution trajectory spans the full task: understanding the
Environment, authoring a Program, reading Feedback, diagnosing behavior,
revising the strategy system, submitting candidates, and completing the final
handoff. These long-horizon records provide training material for coding agents
and policy-engineering agents.

```text
Environment + Benchmark
        │
        ▼
Agent authors and revises a Program
        │
        ├── evolution trajectory ─────▶ Agent SFT / distillation
        └── evaluation outcomes ──────▶ Agent RL / RLVR
```

The Kernel provides the common task, Evaluation, Run, and evidence contracts.
Dataset exporters and training systems can turn retained Agent trajectories
into SFT, RL, and distillation data, then return trained agents for held-out
measurement. In this way, the Environment catalog is both an Agent Benchmark
surface and a scalable source of verifiable long-horizon experience.

## Continue reading

- [Core concepts →](/docs/concepts/)
- [Evaluation and Runs →](/docs/evaluation/)
- [Environment catalog →](/environments/)
- [Core16 results →](/results/)
- [Paper ↗](https://arxiv.org/abs/2607.02440)
