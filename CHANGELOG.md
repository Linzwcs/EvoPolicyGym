# Changelog

## Unreleased

- Added persisted Episode progress events, the public `RunObserver` contract,
  and a standard-library `ConsoleProgress` reporter.

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
