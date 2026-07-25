# EvoPolicyGym Pendulum Benchmark

This directory contains the independently installable Pendulum Benchmark for
EvoPolicyGym. It adapts Gymnasium `Pendulum-v1` through the public EvoPolicyGym
authoring SPI.

The Benchmark contract is:

- one semantic observation dictionary containing `cos_theta`, `sin_theta`,
  and `theta_angular_velocity`;
- one exact finite floating-point torque Action in `[-2.0, 2.0]`;
- one complete Gymnasium Episode with a fixed 200-step horizon;
- deterministic, split-scoped hidden environment seeds;
- mean Episode return as the scalar score;
- `-3300` return for a Policy failure, below the approximately `-3254.72`
  theoretical minimum complete Episode return.

The reward is the negative sum of squared normalized angle error, weighted
angular velocity, and weighted torque. Zero is the maximum per-step reward.
Unlike the other Classic Control tasks, Pendulum has no success termination;
every normal Episode reaches the time limit. The official task contract is
documented in the
[Gymnasium Pendulum reference](https://gymnasium.farama.org/environments/classic_control/pendulum/).

## Public interface

```python
from pendulum import PendulumBenchmark, baseline_program
```

`PendulumBenchmark` owns the public specification, deterministic Episode
planning, fresh Environment construction, scoring, Feedback, and trace
publication. `baseline_program()` returns an intentionally weak zero-torque
Policy suitable as the starting point for development Runs.

Actions are never clipped, cast, or replaced. Integers, booleans, non-finite
floats, out-of-range floats, and structured values raise `InvalidAction`.
The adapter wraps an accepted scalar float in the one-element sequence expected
by Gymnasium.

The Benchmark adapter converts Gymnasium's positional array into:

```json
{
  "cos_theta": -0.996,
  "sin_theta": 0.089,
  "theta_angular_velocity": -0.431
}
```

The names are part of the Benchmark contract. A Policy does not need to infer
the Gymnasium array order.

## Feedback and trace

Feedback reports mean return, mean Episode length, completed and truncated
Episodes, Policy failures, and bounded trace coverage. The public
`trace.jsonl` artifact retains at most eight Episodes. It contains the
observations seen by the Policy, unmodified scalar Actions, rewards, next
observations, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/classic_control/pendulum
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation test is explicitly unsafe and
provides no isolation. The packaged baseline is trusted test code.
