# Changelog

## Unreleased

- Added Benchmark-declared permanent/bulk Artifact retention, complete-newest
  protection, synchronized oldest-first bulk eviction in Host and Agent views,
  Agent-owned `workspace/analysis/`, and `run-record/v5`.
- Added first-class public Environment parameters to `BenchmarkSpec`, the
  `policy/v2` `PolicyContext`, Coding Agent tasks, Evaluation identity, and
  `run-record/v4`.
- Added a canonical Environment-parameter digest so direct Evaluations,
  Submissions, Validation, Assessment, and retained Runs cannot silently mix
  different configured tasks under one Benchmark ID.
- Added optional held-out final-Program Assessment with an independent seed
  domain, aggregate results, progress events, and run-record schema v3.
- Added `AssessmentConfig`, `AssessmentResult`, and the `assessment_failed`
  terminal reason without candidate fallback.
- Added atomic ordered candidate handoff through `agent-session/v2` and
  optional post-Agent server-side Validation with aggregate retained results.
- Added `ValidationConfig`, candidate and Validation fields on `RunResult`,
  the `validation_failed` terminal reason, and run-record schema v2.
- Added persisted Episode progress events, the public `RunObserver` contract,
  and a standard-library `ConsoleProgress` reporter.
- Made the per-Submission Episode cap optional; it now defaults to `None` so
  the Coding Agent can allocate the finite Run budget itself.
- Allowed a Benchmark to publish one bounded Coding Agent skill that
  Program-Evolution Runs may explicitly project read-only into the workspace
  without exposing it to Policy processes.

## 0.3.0

- Replaced the superseded implementation with a small clean-slate Kernel.
- Added immutable Program snapshots and direct per-Episode Policy processes.
- Added bounded Program-Evolution Runs with Agent submissions and final
  selection.
- Added Benchmark-defined Feedback content and public Artifact publication.
- Added the first-party Codex integration for explicitly unsafe local process
  execution.
- Made `evaluation`, `run`, and `execution` cohesive public feature packages;
  removed their parallel private shadow packages and the global composition
  root.
- Added a provider-neutral `CodingAgent` task/invocation template and made
  Codex its first implementation.
- Organized independently installable Benchmark distributions under
  `environments/` and marked the Kernel package as typed for external authors.
- Removed the superseded 0.2 implementation and its experimental products from
  the active repository.
