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

- hull angle in radians plus separately documented normalized hull and linear
  velocities;
- named hip angles in radians, knee angles offset by `+1` radian, and joint
  angular velocities normalized by their respective `4` or `6 rad/s` motor
  speeds;
- exact boolean ground contact for both feet;
- ten normalized lidar range fractions, ordered from downward to forward.

An Action is an exact four-float list in the order
`[left_hip, left_knee, right_hip, right_knee]`. Every value must be in
`[-1.0, 1.0]`. Its sign selects motor direction and its magnitude selects
maximum torque, not motor speed. Hip target speed is `±4 rad/s`, knee target
speed is `±6 rad/s`, and full magnitude permits `80 N·m`. Integers, tuples,
non-finite numbers, and out-of-range values are rejected rather than converted
or clipped.

An Episode terminates when the hull touches the ground, the walker moves behind
the start, or the walker reaches the far end. It otherwise truncates at
Gymnasium's 1600-step limit. On a normal transition:

```text
shaping = 130*world_x/30 - 5*abs(hull_angle)
reward = delta(shaping) - 0.028*sum(abs(action))
```

A hull collision or movement behind the start overrides the whole transition
reward with `-100`; it does not merely add a fall penalty to the normal terms.
The scalar Benchmark score is mean Episode return; Gymnasium's published
solution threshold is 300.

Policy failure receives a conservative `-1000` return. The packaged baseline
applies zero torque and is intentionally weak.

## Feedback and trace

Feedback distinguishes completed courses, falls/movement behind the start, and
time limits. It reports requested versus actually charged motor penalty (the
terminal `-100` override replaces normal terms), forward and posture shaping,
estimated relative course progress, hull stability, normalized forward speed,
foot-contact fractions, action magnitude, lidar clearance, Policy failures,
and bounded trace coverage.

`trace.jsonl` retains at most four Episodes. Transition metrics identify all
four requested motor commands, target speeds and torque caps, reward
decomposition and override status, relative progress, hull pose and recovered
world velocities, contact changes and support phase, nearest lidar ray,
remaining steps, and terminal reason. Episode rows provide matching aggregate
diagnostics and an explicit outcome.

Every Episode with at least one valid transition also preserves every captured
600 × 400 upstream `rgb_array` frame losslessly in `rendered-frames.npz`, with
step indices, rewards, reward presence, and cumulative returns. The Benchmark
captures the initial pose, first result, terminal result, and a fixed stride of
intermediate results, with at most 42 frames per Episode. It derives an H.264
MP4 at five frames per second from exactly those frames for direct playback.
The NPZ is pixel-exact evidence and the MP4 is a presentation artifact; both
use `retention="bulk"`. Raw RGB tensors are removed from `trace.jsonl` and
never become Policy observations. Rendering failure is diagnostic and cannot
change physics or score; zero-step failures publish an explicit unavailable
manifest.

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
