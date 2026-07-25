# EvoPolicyGym Acrobot Benchmark

This directory contains the independently installable Acrobot Benchmark for
EvoPolicyGym. It adapts Gymnasium `Acrobot-v1` through the public
EvoPolicyGym authoring SPI.

The Benchmark contract is:

- one semantic observation dictionary containing six named finite floats: the
  sine and cosine of both joint angles and both angular velocities;
- three exact integer Actions: `0`, `1`, and `2`, applying negative, zero, and
  positive torque;
- one complete Gymnasium Episode with a 500-step hard horizon;
- deterministic, split-scoped hidden environment seeds;
- mean Episode return as the scalar score;
- `-500` return for a Policy failure, matching the minimum complete Episode
  return so invalid code cannot outperform a valid Policy.

Gymnasium assigns `-1` to every non-terminal step and `0` to the successful
terminal step. Higher scores therefore mean reaching the target in fewer
steps. The official task contract is documented in the
[Gymnasium Acrobot reference](https://gymnasium.farama.org/environments/classic_control/acrobot/).

## Public interface

```python
from acrobot import AcrobotBenchmark, baseline_program
```

`AcrobotBenchmark` owns the public specification, deterministic Episode
planning, fresh Environment construction, scoring, Feedback, and trace
publication. `baseline_program()` returns an intentionally weak zero-torque
Policy suitable as the starting point for development Runs.

Actions are never clipped, cast, or replaced. Booleans, floats, out-of-range
integers, and structured values raise `InvalidAction`.

The Benchmark adapter converts Gymnasium's positional array into:

```json
{
  "cos_theta_1": 0.998,
  "sin_theta_1": -0.062,
  "cos_theta_2": 1.000,
  "sin_theta_2": -0.009,
  "theta_1_angular_velocity": -0.100,
  "theta_2_angular_velocity": 0.031
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
cd environments/gymnasium/classic_control/acrobot
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation test is explicitly unsafe and
provides no isolation. The packaged baseline is trusted test code.
