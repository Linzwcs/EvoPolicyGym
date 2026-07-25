# EvoPolicyGym CarRacing Benchmark

This independently installable distribution adapts Gymnasium `CarRacing-v3`
through EvoPolicyGym's public authoring SPI.

## Public interface

```python
from car_racing import CarRacingBenchmark, CarRacingConfig, baseline_program

benchmark = CarRacingBenchmark(
    CarRacingConfig(
        continuous=True,
        lap_complete_percent=0.95,
        domain_randomize=False,
    )
)
```

All dynamics and Action-mode constructor parameters are published through
`BenchmarkSpec.environment_parameters`, contribute to `environment_digest`,
and are delivered to every Policy. The diagnostic-only Gymnasium `verbose`
flag remains fixed to `False`; rendering is Host-owned.

## Contract

The Policy receives the exact native `96 x 96 x 3` RGB frame as a
`TensorValue(dtype="uint8", shape=(96, 96, 3), data=...)`. Pixels are
row-major RGB bytes in `[0, 255]`. No downsampling, cropping, feature
extraction, or color conversion occurs.

In continuous mode, an Action is an exact three-float list
`[steering, gas, brake]`: steering is in `[-1.0, 1.0]`, while gas and brake are
in `[0.0, 1.0]`. In discrete mode, Actions are exact integers:

- `0`: do nothing;
- `1`: steer right;
- `2`: steer left;
- `3`: gas;
- `4`: brake.

Malformed or out-of-range Actions are rejected rather than repaired or
clipped.

An Episode terminates after the configured fraction of the track is completed
or after leaving the playfield, and otherwise truncates at 1000 steps. Reward
is `-0.1` per frame plus `1000/N` for each newly visited tile; leaving the
playfield gives a terminal `-100`. The scalar Benchmark score is mean Episode
return. Gymnasium's published solution threshold is 900.

Policy failure receives a conservative `-1000` return. The continuous packaged
baseline applies half throttle without steering; the discrete baseline applies
full throttle without steering. Both are intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, completed laps, Policy failures, and
bounded trace coverage. `trace.jsonl` retains one complete Episode. It stores
the initial frame and each next frame losslessly as `zlib+base64`, together
with exact Actions, rewards, and termination flags. The encoded object records
the Tensor dtype and shape needed for deterministic reconstruction.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/box2d/car_racing
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

The independent project installs Gymnasium's `box2d` extra. `ProcessExecution`
used by Evaluation tests is explicitly unsafe and provides no isolation. The
packaged baseline is trusted test code.
