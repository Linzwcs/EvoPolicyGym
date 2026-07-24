# Contributing

EvoPolicyGym requires Python 3.12 and uses `uv`.

```console
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
```

The supported package lives under `src/evopolicygym/`. External Benchmark
distributions integrate only through `evopolicygym.authoring`; the active
CartPole example lives under `environments/cartpole/`.

Follow the ownership and import rules in `ARCHITECTURE.md`. Keep concrete I/O
out of pure Evaluation and Program-Evolution rule modules, keep `_protocol`
pure, keep provider-neutral process mechanisms under `execution/process`,
and do not add compatibility namespaces for removed implementations.

`ProcessExecution` is intentionally unsafe for hostile code. Changes involving
runtime semantics must test typed failure behavior and cleanup paths.
