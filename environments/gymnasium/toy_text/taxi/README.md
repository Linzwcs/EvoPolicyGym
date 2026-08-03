# EvoPolicyGym Taxi Benchmark

This independently installable distribution adapts Gymnasium `Taxi-v4`
through EvoPolicyGym's public authoring SPI. A Policy must navigate to a
passenger, pick them up, and deliver them to one of four landmarks.

## Public interface

```python
from taxi import TaxiBenchmark, TaxiConfig, baseline_program

benchmark = TaxiBenchmark(
    TaxiConfig(
        is_rainy=True,
        fickle_passenger=False,
        rainy_probability=0.8,
        fickle_probability=0.3,
    )
)
```

`TaxiConfig` exposes all four non-rendering Gymnasium constructor parameters.
They are published through `BenchmarkSpec.environment_parameters`, contribute
to `environment_digest`, and are delivered to every Policy in
`PolicyContext.environment_parameters`.

In rainy mode, a movement follows the requested direction with
`rainy_probability`; the remaining probability is split between the two
lateral directions. A requested move blocked by a wall or boundary leaves the
taxi in place without lateral drift. With a fickle passenger, reset samples
with `fickle_probability` whether to change the destination once, on the first
successful taxi movement after the first pickup.

## Contract

The positional Gymnasium state is decoded before crossing the Policy boundary:

```json
{
  "state": 341,
  "taxi_row": 3,
  "taxi_column": 2,
  "passenger_location": "red",
  "destination": "green",
  "legal_actions": [0, 1, 3]
}
```

The six exact integer Actions are:

- `0`: move south;
- `1`: move north;
- `2`: move east;
- `3`: move west;
- `4`: pick up the passenger;
- `5`: drop off the passenger.

`legal_actions` is the public Gymnasium advisory mask returned by `reset()` and
`step()`. It lists Actions expected to change the encoded state; it is not an
enforced legality mask. All six integer Actions are accepted. An unlisted wall
movement executes as a `-1` no-op, while an unlisted illegal pickup or dropoff
executes with reward `-10`. Values outside the six-Action integer domain are
never clipped, cast, or replaced.

Every ordinary step and valid pickup costs `-1`. Successful delivery returns
`+20` and terminates the Episode. Dropping an onboard passenger at a wrong
landmark unloads them there and costs the ordinary `-1`; pickup away from the
passenger or dropoff away from a landmark returns `-10` without changing state.
The scalar score is mean Episode return over the requested split. A Policy
failure receives `-2000`, the minimum 200-step complete return.

## Feedback and trace

Feedback reports mean return, mean steps, successful deliveries, time limits,
Policy failures, pickup/dropoff/illegal/no-op event counts, observed destination
changes, and bounded trace coverage. The public `trace.jsonl` artifact retains
at most eight Episodes and includes the complete semantic observations seen by
the Policy, named unmodified Actions, rewards, next observations, termination
flags, advisory-mask membership, event classification, observed movement,
sampled rainy-branch probability, state-component changes, step count, and
terminal reason.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

`baseline_program()` always requests movement south. It is intentionally weak
and leaves navigation, pickup, and delivery strategy to the coding agent.

## Development

From the repository root:

```console
cd environments/gymnasium/toy_text/taxi
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by Evaluation tests is explicitly unsafe and provides
no isolation. The packaged baseline is trusted test code.
