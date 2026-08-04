# EvoPolicyGym robosuite Benchmark

This independent distribution exposes all 19 environments registered by
robosuite 1.5.2 through EvoPolicyGym's public authoring API.

`RobosuiteConfig.profile` selects one fixed task for a Run. Single-arm tasks
use one Panda; two-arm tasks use two opposed Pandas. All profiles use the
upstream BASIC composite controller with OSC pose control, state observations,
upstream dense shaping where implemented, and success-rate scoring.

```python
from robosuite_benchmarks import RobosuiteBenchmark, RobosuiteConfig

benchmark = RobosuiteBenchmark(
    RobosuiteConfig(profile="lift", max_episode_steps=500)
)
```

## Profiles

- Lift, Stack, Door, Wipe, and ToolHang;
- NutAssembly plus Single, Square, and Round variants;
- PickPlace plus Single, Milk, Bread, Cereal, and Can variants;
- TwoArmLift, TwoArmPegInHole, TwoArmHandover, and TwoArmTransport.

The Policy observation contains two canonical `float64` tensors:
`proprioception` concatenates the upstream per-robot proprioceptive state and
`objects` contains the task's public object state. Camera frames, simulator
objects, Host paths, seeds, and raw Case identity never cross the Policy
boundary.

Actions are strict BASIC-controller vectors in `[-1, 1]`. Single-arm tasks use
six operational-space pose deltas and, where a gripper exists, one gripper
effort. Two-arm tasks concatenate one such control block per robot. Invalid
Actions are rejected without advancing MuJoCo; they are never clipped or
repaired.

An Episode succeeds if the upstream task success predicate is true on any
transition. The Environment continues until the configured horizon unless the
upstream task declares an earlier terminal condition. Feedback reports success,
dense return, first success, action magnitude/saturation, state motion, cleanup
outcomes, and bounded public transition traces.

## Runtime pin

The distribution pins `robosuite==1.5.2` and the independently tested
`mujoco>=3.3,<3.4` ABI. Current unconstrained resolution reaches MuJoCo 3.11,
which removed the `MjData.qM` attribute still used by robosuite 1.5.2's
operational-space controller and prevents every task from constructing. This
pin should be revisited when robosuite adopts the current API.

## Development

```console
uv sync --extra dev
uv run ruff check src tests scripts
uv run mypy --no-incremental
uv run python -m unittest discover -s tests
uv build
```

The optional Luna end-to-end check uses the explicitly unsafe local process
execution profile:

```console
uv run python scripts/run_robosuite_codex.py \
  --record-to /private/tmp/robosuite-lift-luna \
  --model gpt-5.6-luna \
  --reasoning-effort high \
  --allow-unsafe-process
```

`ProcessExecution` is not a sandbox. Run the command only with trusted Agent
and Policy code. A short record path avoids the macOS Unix-domain socket limit;
move the closed Run record under the repository's `runs/` directory afterward.
