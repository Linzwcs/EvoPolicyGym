---
name: use-evopolicygym
description: Set up, run, extend, and diagnose EvoPolicyGym policy-evolution workflows. Use when installing EvoPolicyGym or an Environment distribution; creating or evaluating a Policy Program; running Codex or another Coding Agent against a Benchmark; inspecting Feedback, Artifacts, progress, or Run records; implementing an external Benchmark or Environment package; or troubleshooting submission, evaluation, publication, budget, and terminal-state failures.
---

# Use EvoPolicyGym

Use the public EvoPolicyGym SDK to build a complete bounded loop:

```text
Program → Evaluation → Feedback → Agent edit → Submission → candidate handoff
→ Host selection → held-out Assessment
```

Preserve the separation between the trusted Host and Benchmark, the Coding
Agent workspace, and the fresh Policy process created for every Episode.

## Choose the workflow

1. **Evaluate a Program:** read [references/api.md](references/api.md), capture
   a Policy directory as a `Program`, and call `evaluate()`.
2. **Run a Coding Agent:** read [references/api.md](references/api.md), choose
   finite submission and Episode budgets, create a new Run directory, and call
   `run()`.
3. **Author or modify a Benchmark:** read
   [references/authoring.md](references/authoring.md) completely before editing
   an Environment distribution.
4. **Diagnose a completed or failed Run:** read
   [references/diagnostics.md](references/diagnostics.md) and inspect retained
   Host records without editing them.
5. **Change the Kernel itself:** first read the repository `AGENTS.md` and
   `ARCHITECTURE.md`; preserve their ownership and import rules.

If the task spans multiple workflows, perform authoring or setup first, direct
Evaluation second, and a Coding Agent Run last.

## Establish the local context

- Confirm Python 3.12 and the installed EvoPolicyGym version.
- In this repository, use `uv` and prefer the active source under
  `src/evopolicygym/`; ignore `reference/` unless the user explicitly requests
  historical repair.
- Treat every package under `environments/` as an independently installable
  Benchmark distribution. Do not assume it is included in the Kernel wheel.
- Inspect the selected Environment package README and public imports before
  constructing commands. Do not guess a Benchmark class, baseline factory, or
  extra dependency.
- Use only public modules documented by the selected release. External
  packages may use `evopolicygym.authoring`, but never private `_...` modules.

## Apply the safety gate

`ProcessExecution.unsafe()` is an acknowledgement, not isolation. The Agent
and submitted Policy execute with the current operating-system user's
authority.

- State this limitation before running code that is not already trusted.
- Obtain any execution approval required by the current tool or user context.
- Do not claim that a virtual environment, subprocess, workspace, or
  `--allow-unsafe-process` flag is a sandbox.
- Do not silently substitute local processes when formal virtualization was
  requested. Current 0.3 does not provide whole-Run virtualization.

## Build a valid Policy Program

- Place `policy.py` in the Program directory.
- Export `make_policy(context: PolicyContext)`.
- Return an object with `act(observation: PolicyValue) -> PolicyValue`.
- Use only the bounded PolicyValue ABI across the Policy boundary.
- Read public task configuration from
  `PolicyContext.environment_parameters`; never expect private Episode
  scenarios or Environment seeds there.
- Never expose Host paths, descriptors, credentials, private Episode seeds,
  Case identity, scorer objects, or custom Python objects to the Policy.
- Expect a fresh process, Policy instance, and scratch directory for every
  Episode. Persist state only between `act()` calls in the same Episode.
- Return exact Actions. The Kernel does not repair invalid Actions.

## Run a bounded loop

- Use direct Evaluation to validate imports, termination, legal Actions, and
  cleanup before spending a Coding Agent budget.
- Set explicit finite budgets. For trace-heavy Benchmarks, prefer an explicit
  `max_episodes_per_submission` documented as safe by that Benchmark.
- Use a new `record_to` path whose parent already exists. Never reuse or
  pre-create the Run directory.
- Enable `use_benchmark_skill=True` only when the Benchmark supplies useful
  task-specific guidance. That read-only skill complements this project skill;
  it does not replace the Host task or expose data to the Policy.
- Add `ConsoleProgress()` when interactive progress is useful.
- Require the Agent to publish candidates through
  `evopolicygym submit program --episodes N` and finish by handing the Host
  one or more published IDs with
  `evopolicygym finish SUBMISSION_ID [SUBMISSION_ID ...]`.
- Use `ValidationConfig` when the Host should compare multiple candidates.
  Give every candidate the same split, derived seed, Episode count, and Policy
  seeds. Its Episode allocation is separate from the Agent search budget.
- Treat successful `finish` as an authority boundary. Validation and final
  selection occur only after the Agent process is reaped; do not expose their
  evidence in workspace Feedback.
- Use `AssessmentConfig` to measure only the selected Program on a separately
  seeded held-out split. Assessment must not rerank candidates, publish
  evidence to the Agent, retry automatically, or fall back after a trusted
  failure.
- Treat failed trusted evaluation attempts as consumed budget. Never rewrite
  accounting after a failure.

## Handle Feedback correctly

- Assume only that Feedback has a finite scalar score, Benchmark-defined
  public content, and zero or more bounded public Artifacts.
- Read `feedback/latest.json`, then follow its relative Artifact paths.
- Let the Benchmark define trace, diagnostics, replay, image, or report
  formats. Do not impose a Kernel-wide trace schema.
- Keep scoring based on every evaluated Episode even when a Benchmark retains
  only a bounded sample of detailed traces.
- Do not infer hidden seeds, pool identity, or future Environment state from
  public Feedback.

## Verify proportionally

For Kernel changes, run:

```console
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build
```

For an Environment distribution, run its own Ruff, mypy, unittest, and build
commands in that package. Test deterministic Episode planning, cleanup,
Policy-failure classification, Feedback privacy, and Artifact bounds.

Report the selected Benchmark, Program digest or source, split, seed, Episode
budget, ordered candidate IDs, Validation configuration and aggregate result,
Assessment configuration and aggregate result, terminal reason, final
submission, verification commands, and the remaining lack of process
isolation.
