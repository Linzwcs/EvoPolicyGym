# EvoPolicyGym FrozenLake Benchmark

This independently installable distribution adapts Gymnasium FrozenLake
through EvoPolicyGym's public authoring SPI. It is the first Gymnasium
Benchmark in this repository whose environment is configured explicitly at
Benchmark construction time.

## Public interface

```python
from frozen_lake import (
    FrozenLakeBenchmark,
    FrozenLakeConfig,
    baseline_program,
)

benchmark = FrozenLakeBenchmark(
    FrozenLakeConfig(
        map_name="8x8",
        is_slippery=True,
        success_rate=1.0 / 3.0,
    )
)
```

`FrozenLakeConfig` accepts the upstream standard `4x4` and `8x8` maps, an
`is_slippery` switch, and a finite `success_rate` in `[0.0, 1.0]`. The
configuration is published through `BenchmarkSpec.environment_parameters` and
contributes to `environment_digest`. Every Policy instance receives a detached
copy in `PolicyContext.environment_parameters`.

Environment parameters are Case-independent and remain constant across a
Run. `EpisodeSpec` contains only the hidden, split-scoped environment seed;
using `EpisodeSpec.scenario` for map or dynamics configuration is rejected.

## Contract

The Policy observation is semantic rather than a bare Gymnasium integer:

```json
{
  "state": 0,
  "row": 0,
  "column": 0,
  "tile": "S"
}
```

Actions are exact integers: `0` moves left, `1` down, `2` right, and `3` up.
When the lake is slippery, the requested direction succeeds with
`success_rate`; each adjacent direction receives half of the remaining
probability.

Reaching the goal returns `1` and terminates the Episode. Frozen tiles and
holes return `0`; a hole terminates the Episode. The scalar Benchmark score is
the fraction of Episodes that reach the goal. The standard 4×4 map has a
100-step horizon and the standard 8×8 map has a 200-step horizon.

`baseline_program()` performs value iteration using only the map and dynamics
published to the Policy. It does not receive private Case identity or hidden
seeds.

## Feedback and trace

Feedback reports success rate, mean return, mean steps, Policy failures, and
bounded trace coverage. The public `trace.jsonl` artifact retains at most eight
Episodes and includes semantic observations, Actions, rewards, next
observations, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/toy_text/frozen_lake
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
