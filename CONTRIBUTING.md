# Contributing

Contributions of all sizes are welcome, including new interactive environment
integrations, Kernel improvements, documentation, tests, and representative
test data.

## Contribute with coding agents

We recommend using a coding agent to contribute to EvoPolicyGym. Claude Code
and Codex can use the repository's
[`evopolicygym` Agent Skill](skills/evopolicygym/) for project-specific
guidance on Benchmark authoring, provider integration, Run diagnostics, and
Kernel development. Ask the agent to read the Skill and the repository
instructions before it starts making changes.

## Development setup

EvoPolicyGym requires Python 3.12 and uses `uv`.

```console
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
```

## Project boundaries

The supported package lives under `src/evopolicygym/`. External Benchmark
distributions integrate only through `evopolicygym.authoring`; the active
CartPole example lives under
`environments/gymnasium/classic_control/cartpole/`.

Follow the ownership and import rules in `ARCHITECTURE.md`. Keep concrete I/O
out of pure Evaluation and Program-Evolution rule modules, keep `_protocol`
pure, keep provider-neutral process mechanisms under `execution/process`,
and do not add compatibility namespaces for removed implementations.

`ProcessExecution` is intentionally unsafe for hostile code. Changes involving
runtime semantics must test typed failure behavior and cleanup paths.

## Contact

To propose an environment integration, coordinate a larger change, or ask
where to start, open a GitHub issue or email
[zhilin.nlp@gmail.com](mailto:zhilin.nlp@gmail.com).
