# MiniGrid Memory Benchmark

An independently installable EvoPolicyGym Benchmark for the
[MiniGrid Memory](https://minigrid.farama.org/environments/minigrid/MemoryEnv/)
partially observable T-maze.

The Policy sees the upstream `7 × 7 × 3` egocentric symbolic image, compass
direction, and mission text. It must remember whether the green cue was a key
or a ball, walk down the corridor, and select the matching object. The primary
score is Episode success rate.

## Install and test

From the EvoPolicyGym repository root:

```console
uv sync --project environments/minigrid/minigrid/memory --extra dev
uv run --project environments/minigrid/minigrid/memory \
  python -m unittest discover \
  -s environments/minigrid/minigrid/memory/tests
uv build environments/minigrid/minigrid/memory
```

## Public API

```python
from minigrid_memory import MemoryBenchmark, MemoryConfig, baseline_program

benchmark = MemoryBenchmark(
    MemoryConfig(profile="13x13-random"),
)
program = baseline_program()
```

Available profiles are `11x11`, `13x13`, `13x13-random`, and
`17x17-random`. The default random-length profile prevents a Policy from
solving the task using one fixed action count.

Feedback includes aggregate success, return, step, wrong-target, truncation,
and Policy-failure statistics. `trace.jsonl` contains a bounded semantic trace
for at most four Episodes; long traces retain the first 128 and last 32
transitions. Trace rows and visible-object records are derived only from
Policy-visible observations and contain no Episode seed or private Case
identity.

The packaged baseline is intentionally small. It is a finite-state Policy
whose same-Episode state remembers the cue; it uses only public Policy context
and observations.

Tests that use `ProcessExecution.unsafe()` run trusted packaged code only.
That backend is a local process mechanism, not a sandbox and not suitable for
hostile Programs.
