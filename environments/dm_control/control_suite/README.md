# EvoPolicyGym DeepMind Control Suite Benchmark

This independent distribution exposes all 28 tasks in dm-control 1.0.43's
official `suite.BENCHMARKING` collection through EvoPolicyGym's public
authoring API.

`DmControlConfig.profile` fixes one domain/task pair for a Run. Every profile
uses the upstream state observations, continuous task reward, action bounds,
task randomization, control timestep, and 1,000-step default time limit.

```python
from dm_control_benchmarks import DmControlBenchmark, DmControlConfig

benchmark = DmControlBenchmark(
    DmControlConfig(profile="cartpole-swingup", max_episode_steps=1_000)
)
```

## Coverage

- Acrobot (2), Ball in Cup (1), Cartpole (4), Cheetah (1), Finger (3),
  Pendulum (1), Point Mass (1), and Reacher (2);
- Fish (2), Hopper (2), Humanoid (3), Swimmer (2), and Walker (3);
- Manipulator Bring Ball (1).

The Policy observes a dictionary of named `float64` tensors that preserves the
upstream observation fields and shapes. Simulator objects, render contexts,
Host paths, seeds, and raw Episode identity never cross the Policy boundary.
These values are `TensorValue` objects rather than indexable sequences; their
`data` is packed little-endian `float64`, and the packaged baseline shows a
standard-library-only `struct.iter_unpack("<d", ...)` decoder.

Actions are exact-float arrays with the upstream shape and bounds. For every
current benchmarking task the bounds are `[-1, 1]`. Invalid Actions are
rejected before `Environment.step()`; they are never clipped or repaired.
Feedback scores mean continuous return and provides bounded public JSONL
traces plus reward, discount, action, and state-motion diagnostics. For every
Episode with at least one valid transition, it also preserves every captured
128x128 frame from the suite's `camera_id=0` losslessly in
`rendered-frames.npz`. The evidence contains frame pixels, step indices,
rewards, reward presence, and cumulative returns. A replay GIF is derived from
the same frames for convenient inspection; it is not the sole visual evidence.

The Host captures the initial state, step 1, a fixed stride chosen from the
configured horizon, and the terminal state, for at most 42 frames per Episode.
Short Episodes capture every step. Raw RGB values exist in Host-only Step
metrics during evaluation, are removed from the JSONL trace, and never become
Policy observations. The lossless frame evidence, derived GIF, and
reproducible trace all use `retention="bulk"`. Each visual-evidence manifest
states whether capture is complete for the configured sampling schedule and
whether it covers every Episode step. The Kernel-owned `feedback.json`
`episodes` array preserves every Episode's status, reward, step count, and
Policy failure without summary truncation. A zero-step Policy failure is
recorded there with an explicit unavailable visual-evidence manifest because
the public Environment SPI has no post-reset artifact channel.

Camera capture is diagnostic evidence and is fail-soft: a missing offscreen GL
backend is reported in Feedback without changing the physics Episode or score.
On macOS the adapter uses MuJoCo's native CGL renderer so Episode worker
threads do not enter dm_control's GLFW window path. Headless Linux deployments
should provision a working MuJoCo EGL or OSMesa profile.

## Runtime pin

The distribution pins `dm-control==1.0.43` and `mujoco>=3.10,<3.11`.
dm-control 1.0.43's generated MuJoCo structure indexer reads `MjData.qM`, which
is present in MuJoCo 3.10 but absent in 3.11. The unconstrained dependency
currently resolves to 3.11 and fails while constructing the first task. Revisit
the upper bound when dm-control publishes generated bindings for that layout.

## Development

```console
uv sync --extra dev
uv run ruff check src tests scripts
uv run mypy --no-incremental
uv run python -m unittest discover -s tests
uv build
```

The Luna end-to-end check uses the explicitly unsafe local process execution
profile:

```console
uv run python scripts/run_dm_control_codex.py \
  --record-to /private/tmp/dmc-luna \
  --model gpt-5.6-luna \
  --reasoning-effort high \
  --allow-unsafe-process
```

`ProcessExecution` is not a sandbox. Run this only with trusted Agent and
Policy code. A short record path avoids the macOS Unix-domain socket limit;
move the closed Run record under the repository's `runs/` directory afterward.
