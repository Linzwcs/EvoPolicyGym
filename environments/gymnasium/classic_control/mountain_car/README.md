# EvoPolicyGym Mountain Car Benchmark

This directory contains the independently installable Mountain Car Benchmark
for EvoPolicyGym. It adapts Gymnasium `MountainCar-v0` through the public
EvoPolicyGym authoring SPI.

The Benchmark contract is:

- one semantic observation dictionary containing the car's finite `position`
  and signed `velocity`, with units and meanings in the public specification;
- three exact integer Actions: `0` accelerates left, `1` applies no
  acceleration, and `2` accelerates right;
- one complete Gymnasium Episode with a 200-step hard horizon;
- deterministic, split-scoped hidden environment seeds;
- mean Episode return as the scalar score;
- `-200` return for a Policy failure, matching the minimum complete Episode
  return so invalid code cannot outperform a valid Policy.

Gymnasium assigns `-1` to every transition, including the successful one.
Higher scores therefore mean reaching position `0.5` in fewer steps. The
initial position is sampled from `[-0.6, -0.4]`, while initial velocity is zero.

The complete public dynamics are also carried by `BenchmarkSpec`. For action
`a` in `{0, 1, 2}`, the adapter checks Gymnasium against:

```text
velocity' = clip(velocity + (a - 1)*0.001 - cos(3*position)*0.0025,
                 -0.07, 0.07)
position' = clip(position + velocity', -1.2, 0.6)
```

Moving left into position `-1.2` resets velocity to zero. A successful Episode
requires position at least `0.5` and nonnegative velocity. The official task
contract is documented in the
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

Feedback reports mean return, mean Episode length, success and time-limit
counts, mean per-Episode minimum and maximum positions, the overall furthest
position, closest remaining goal gap, Policy failures, and bounded trace
coverage. These progress values distinguish a Policy that builds useful
momentum from one that simply survives for the same `-200` return.

The public `trace.jsonl` artifact retains at most eight Episodes. Every
transition contains the observations seen by the Policy, unmodified Action and
its meaning, reward, termination flags, and public transition metrics. Metrics
include engine and gravity velocity increments, state deltas, direction
reversals, left-wall collisions, terrain height, goal distance, running
position extrema, remaining steps, and the terminal reason. Episode rows state
the outcome and trajectory extrema.

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
