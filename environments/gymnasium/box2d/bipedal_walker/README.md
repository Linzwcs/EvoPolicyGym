# EvoPolicyGym BipedalWalker Benchmark

This independently installable distribution adapts Gymnasium
`BipedalWalker-v3` through EvoPolicyGym's public authoring SPI.

## Public interface

```python
from bipedal_walker import (
    BipedalWalkerBenchmark,
    BipedalWalkerConfig,
    baseline_program,
)

benchmark = BipedalWalkerBenchmark(
    BipedalWalkerConfig(hardcore=False)
)
```

The non-rendering Gymnasium constructor parameter `hardcore` is published
through `BenchmarkSpec.environment_parameters`, contributes to
`environment_digest`, and is delivered to every Policy. Hardcore terrain adds
stumps, pits, and stairs.

## Contract

The Policy receives a semantic object containing:

- normalized hull angle, angular velocity, and horizontal/vertical velocity;
- named hip and knee angles and angular velocities for both legs;
- exact boolean ground contact for both feet;
- ten normalized lidar range fractions, ordered from downward to forward.

An Action is an exact four-float list in the order
`[left_hip, left_knee, right_hip, right_knee]`. Every value must be in
`[-1.0, 1.0]`. Its sign selects motor direction and its magnitude selects
torque. Integers, tuples, non-finite numbers, and out-of-range values are
rejected rather than converted or clipped.

An Episode terminates when the hull touches the ground, the walker moves behind
the start, or the walker reaches the far end. It otherwise truncates at
Gymnasium's 1600-step limit. Reward measures forward progress and upright
posture, subtracts motor energy, and gives `-100` on a fall. The scalar
Benchmark score is mean Episode return; Gymnasium's published solution
threshold is 300.

Policy failure receives a conservative `-1000` return. The packaged baseline
applies zero torque and is intentionally weak.

## Feedback and trace

Feedback reports mean return, mean steps, completed courses, Policy failures,
and bounded trace coverage. `trace.jsonl` retains at most four Episodes with
complete semantic observations, exact Actions, rewards, and termination flags.

Environment seeds, Policy seeds, Host paths, credentials, and private runtime
evidence are never published.

## Development

From the repository root:

```console
cd environments/gymnasium/box2d/bipedal_walker
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

The independent project installs Gymnasium's `box2d` extra. `ProcessExecution`
used by Evaluation tests is explicitly unsafe and provides no isolation. The
packaged baseline is trusted test code.
