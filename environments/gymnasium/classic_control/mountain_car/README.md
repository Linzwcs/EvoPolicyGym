# EvoPolicyGym Mountain Car Benchmark

This directory contains the independently installable Mountain Car Benchmark
for EvoPolicyGym. It adapts Gymnasium `MountainCar-v0` through the public
EvoPolicyGym authoring SPI.

The Benchmark contract is:

- one semantic observation dictionary containing the car's finite `position`
  and `velocity`;
- three exact integer Actions: `0` accelerates left, `1` applies no
  acceleration, and `2` accelerates right;
- one complete Gymnasium Episode with a 200-step hard horizon;
- deterministic, split-scoped hidden environment seeds;
- mean Episode return as the scalar score;
- `-200` return for a Policy failure, matching the minimum complete Episode
  return so invalid code cannot outperform a valid Policy.

Gymnasium assigns `-1` to every step. Higher scores therefore mean reaching
position `0.5` in fewer steps. The initial position is sampled from
`[-0.6, -0.4]`, while the initial velocity is zero. The official task contract
is documented in the
[Gymnasium Mountain Car reference](https://gymnasium.farama.org/environments/classic_control/mountain_car/).

## Public interface

```python
from mountain_car import MountainCarBenchmark, baseline_program
```

`MountainCarBenchmark` owns the public specification, deterministic Episode
planning, fresh Environment construction, scoring, Feedback, and trace
publication. `baseline_program()` returns an intentionally weak no-acceleration
Policy suitable as the starting point for development Runs.

Actions are never clipped, cast, or replaced. Booleans, floats, out-of-range
integers, and structured values raise `InvalidAction`.

The Benchmark adapter converts Gymnasium's positional array into:

```json
{
  "position": -0.527,
  "velocity": 0.004
}
```

The names are part of the Benchmark contract. A Policy does not need to infer
the Gymnasium array order.

## Feedback and trace

Feedback reports mean return, mean Episode length, successful Episodes, Policy
failures, and bounded trace coverage. The public `trace.jsonl` artifact retains
at most eight Episodes. It contains the observations seen by the Policy,
unmodified Actions, rewards, next observations, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/classic_control/mountain_car
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation test is explicitly unsafe and
provides no isolation. The packaged baseline is trusted test code.
