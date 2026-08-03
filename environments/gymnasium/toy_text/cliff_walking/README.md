# EvoPolicyGym CliffWalking Benchmark

This independently installable distribution adapts Gymnasium
`CliffWalking-v1` through EvoPolicyGym's public authoring SPI. The Policy must
cross a 4×12 grid from start to goal while avoiding the cliff between them.

## Public interface

```python
from cliff_walking import (
    CliffWalkingBenchmark,
    CliffWalkingConfig,
    baseline_program,
)

benchmark = CliffWalkingBenchmark(
    CliffWalkingConfig(is_slippery=False)
)
```

`CliffWalkingConfig.is_slippery` is published through
`BenchmarkSpec.environment_parameters`, contributes to `environment_digest`,
and is delivered to every Policy. In slippery mode, the requested direction
and its two adjacent directions each occur with probability `1/3`. The full
map, start, goal, cliff coordinates, state encoding, edge behavior, rewards,
and horizon are also public environment parameters.

## Contract

The positional Gymnasium state is converted into:

```json
{
  "state": 36,
  "row": 3,
  "column": 0,
  "tile": "start"
}
```

Actions are exact integers: `0` moves up, `1` right, `2` down, and `3` left.
An ordinary step returns `-1`. Entering any cliff cell returns `-100` and
places the player back at the start without terminating. Reaching the goal
returns `-1` and terminates the Episode.

The cliff landing and return to start are atomic: the live observation never
contains `tile="cliff"`. A `-100` transition reports an explicit `cliff_fall`
event and `cliff_then_reset_to_start` movement in feedback, avoiding the false
appearance of an unexplained no-op at the start.

Gymnasium 1.3.0 does not register a TimeLimit for CliffWalking. This Benchmark
adds an explicit 200-step horizon in its adapter and returns a normal truncated
step when the horizon is reached. This makes non-progressing but valid Policies
scoreable rather than treating them as Environment faults.

The scalar score is mean Episode return. Policy failure receives `-20000`,
equal to the minimum 200-step complete return.

## Feedback and trace

Feedback reports mean return, mean steps, successful goal reaches, cliff falls,
boundary no-ops, time limits, Policy failures, and bounded trace coverage.
`trace.jsonl` retains at most eight Episodes with complete semantic
observations, named unmodified Actions, rewards, termination flags, requested
and observed movement, event classification, sampled branch probability,
possible sampled directions, total probability of the observed outcome, step
count, and terminal reason.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

`baseline_program()` always moves right, repeatedly entering the cliff from
the initial state. It intentionally establishes the minimum valid baseline.

## Development

From the repository root:

```console
cd environments/gymnasium/toy_text/cliff_walking
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
