---
locale: en
page: concepts
section: core
title: "Core concepts"
navTitle: "Core concepts"
description: "The domain model and trust boundaries of EvoPolicyGym 0.3."
lead: "Programs are immutable, Evaluations are bounded, Feedback is committed, and each Episode receives a fresh Policy lifecycle."
index: D2
order: 2
docsVersion: v0.3
status: draft
---

## Domain vocabulary

| Value | Meaning |
| --- | --- |
| `Program` | A detached, immutable, content-addressed snapshot of one Policy source directory. |
| `Episode` | One trusted scenario, one fresh Environment, and one fresh Policy process and instance. |
| `Evaluation` | One Program evaluated over a finite deterministic Episode plan. |
| `Feedback` | A Benchmark-defined public projection with one scalar score, bounded content, and optional artifacts. |
| `Submission` | One Program and the committed Feedback produced when a Coding Agent requests Evaluation. |
| `ProgramEvolutionRun` | One bounded outer loop in which a Coding Agent edits Programs, submits candidates, reads Feedback, and hands published candidates to Host-side selection. |
| `Experiment` | Reserved for a future collection of comparable Runs. |

The public SDK uses `Program`, not `ProgramVersion`. A Program retains no Host
source path and cannot change when the caller later edits its original
directory.

## Evaluation lifecycle

```text
Program
  ↓
deterministic Episode plan
  ↓
fresh Environment + fresh Policy process
  ↓
unmodified Actions and trusted Steps
  ↓
sanitized Episode summaries
  ↓
Benchmark-defined Feedback
```

Policy state may persist between `act()` calls in one Episode. It never
persists into another Episode. Cross-Episode improvement happens only when the
outer Coding Agent authors a new Program.

## Program-evolution lifecycle

```text
initial Program
  ↓
Host fixes indexed training Episode pool
  ↓
Coding Agent edits workspace/program/
  ↓
Submission(selector) → fresh runtimes → committed Feedback
  ↓
Coding Agent reads indexed outcomes in workspace/feedback/
  ↓
next Program or finish(candidate IDs)
  ↓
Host-side final selection
```

A `RunConfig` fixes the split, maximum submissions, total Episode budget,
fixed training Episode-pool size, optional per-Submission Episode cap, seed,
and timeouts before the Agent starts. Pool size and budget are different
limits: the pool controls how many Episode identities are available, while the
budget controls the total number of selected indices across all Submissions.
The pool size defaults to the total budget.

The Agent selects public Run-local indices from that pool. The same index
preserves its hidden Episode specification and Policy seed across Submissions,
which supports matched Program comparisons. Every use still creates fresh
runtime state and consumes budget again. Pool indices are experimental handles,
not Environment seeds, and neither the Agent nor the Policy receives the
underlying seed. The optional per-Submission cap defaults to `None`.

## Trust boundary

| Trusted Host and Benchmark own | Policy can observe |
| --- | --- |
| Environment parameter selection | Public `environment_parameters` fixed before Evaluation |
| Episode scenario, Environment seed, and pool mapping | `PolicyContext` without a pool index or Case identity |
| Environment state and transitions | Public observations |
| Action validation | Its own Episode-local state |
| Rewards, scoring, and private metrics | Committed public Feedback only |
| Run budget and publication | No Host path, credential, scorer, or runtime evidence |

The Policy boundary carries only bounded `PolicyValue` data. Paths, file
descriptors, credentials, arbitrary Python objects, and pickle graphs never
cross it.

## Failure ownership

Policy exceptions, timeouts, protocol errors, and invalid Actions become
sanitized Policy failures. Invalid Actions are never clipped, repaired,
sampled, or replaced.

Trusted Environment, Benchmark, process-control, and cleanup faults abort the
Evaluation. They never become Policy penalties.

## Package boundaries

The base `evopolicygym` wheel owns the portable Kernel. Independent Benchmark
distributions depend only on public SDK facades and `evopolicygym.authoring`.
The Kernel does not import those distributions.

The optional Firecracker groundwork is a separate product and does not make a
formal or isolated execution profile available.

## Next

- [Policy ABI →](./policy.md)
- [Evaluation and Runs →](./evaluation.md)
- [Benchmark authoring →](./authoring.md)
- [Execution and safety →](./runtime.md)
