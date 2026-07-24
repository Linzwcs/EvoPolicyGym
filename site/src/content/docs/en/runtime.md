---
locale: en
page: runtime
section: core
title: "Execution and safety"
navTitle: "Execution and safety"
description: "ProcessExecution lifecycle guarantees, failure ownership, and isolation limits."
lead: "Fresh Episode lifecycle separation is implemented. Hostile-code containment is not."
index: D5
order: 5
docsVersion: v0.3
status: draft
---

## Explicit unsafe selection

Local evaluation requires an explicit acknowledgement:

```python
from evopolicygym.execution import ProcessExecution

execution = ProcessExecution.unsafe()
```

This selects local process execution. It does not activate a sandbox, remove
permissions, or make untrusted code safe.

## Lifecycle guarantees

For every Episode, the evaluator:

1. creates a fresh Benchmark-owned Environment;
2. materializes the immutable Program into fresh scratch;
3. starts a fresh Python Policy process;
4. calls `make_policy(context)` once;
5. preserves that instance across the Episode's `act()` calls;
6. validates and applies complete, unmodified Actions;
7. closes the Environment and reaps the Policy process on every exit path.

Policy state and scratch never intentionally cross Episode boundaries.

## Isolation limits

`ProcessExecution` does not provide:

- kernel, namespace, seccomp, cgroup, container, or microVM isolation;
- CPU, memory, PID, descriptor, disk, or network confinement;
- protection for Host files, credentials, processes, or system time;
- adversarial execution of third-party Agent or Policy code;
- bit-for-bit determinism for arbitrary Python Programs.

The Agent process is also local and unisolated. In the current Codex
integration it has the authority of the current operating-system user.

## Failure domains

| Failure | Result |
| --- | --- |
| Policy exception, timeout, invalid Action, or protocol error | Sanitized `policy_failed` Episode; Evaluation may continue with its plan. |
| Environment reset, step, scoring, or cleanup fault | Abort Evaluation; never become a Policy penalty. |
| Process preparation, control, framing, or cleanup fault | Abort Evaluation as an execution fault. |
| Run publication or projection fault | Close the Run; never forge committed Feedback. |

Complete malformed guest frames, partial frames, and trusted-input errors keep
distinct internal classifications even when public Policy feedback remains
sanitized.

## Optional native products

The separately built `native/bootstrap` distribution contains formal-manager
ownership primitives. Its presence alone does not make a formal profile
available.

The separately installable `policy-backend-firecracker` package is alpha
groundwork. Installing or building it does not provide a sandbox, production
backend, or qualified formal execution profile.

No requested virtualization setting may silently fall back to
`ProcessExecution`.

## Operational guidance

Use local process execution only when:

- the Agent, Program, and Benchmark are trusted;
- the caller accepts Host-level authority;
- retained Run files contain no credentials;
- evaluation is for development or conformance, not adversarial admission.

## Next

- [Evaluation and Runs →](../evaluation/)
- [Benchmark authoring →](../authoring/)
- [Architecture notes →](../../research/)
