# EvoPolicyGym Continuous Mountain Car Benchmark

This directory contains the independently installable Continuous Mountain Car
Benchmark for EvoPolicyGym. It adapts Gymnasium
`MountainCarContinuous-v0` through the public EvoPolicyGym authoring SPI.

The Benchmark contract is:

- one semantic observation dictionary containing the car's finite `position`
  and `velocity`;
- one exact finite floating-point Action in `[-1.0, 1.0]`, representing
  directional force;
- one complete Gymnasium Episode with a 999-step hard horizon;
- deterministic, split-scoped hidden environment seeds;
- mean Episode return as the scalar score;
- `-100` return for a Policy failure, below the minimum theoretical complete
  Episode return of `-99.9`.

Each step subtracts `0.1 × action²`, and reaching position `0.45` adds 100
points. Higher scores therefore reward reaching the goal with less control
effort. The official task contract is documented in the
[Gymnasium Continuous Mountain Car reference](https://gymnasium.farama.org/main/environments/classic_control/mountain_car_continuous/).

## Public interface

```python
from mountain_car_continuous import (
    MountainCarContinuousBenchmark,
    baseline_program,
)
```

`MountainCarContinuousBenchmark` owns the public specification, deterministic
Episode planning, fresh Environment construction, scoring, Feedback, and trace
publication. `baseline_program()` returns an intentionally weak zero-force
Policy suitable as the starting point for development Runs.

Actions are never clipped, cast, or replaced. Integers, booleans, non-finite
floats, out-of-range floats, and structured values raise `InvalidAction`.
The adapter wraps an accepted scalar float in the one-element sequence expected
by Gymnasium.

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
unmodified scalar Actions, rewards, next observations, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/classic_control/mountain_car_continuous
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation test is explicitly unsafe and
provides no isolation. The packaged baseline is trusted test code.
