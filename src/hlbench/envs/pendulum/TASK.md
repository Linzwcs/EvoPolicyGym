# Pendulum-v1 — Swing-Up and Stabilize

## Goal

Swing a frictionless pendulum to the upright position (angle = 0) and
keep it balanced there with minimum control effort, starting from a
random initial state each episode.

## The `Policy` you write

A starter at `workspace/system/policy.py` has been auto-staged for
you. Edit it; the class name `Policy` and the three method signatures
below MUST stay exactly the same — the harness imports `Policy` from
`policy.py` once per submit and reuses the instance across the
submit's episodes.

### Required interface

```python
class Policy:
    def __init__(self,
                 obs_space: dict,
                 action_space: dict,
                 env_meta: dict) -> None: ...

    def reset(self, episode_index: int) -> None: ...

    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

### `act(obs)` — the per-step contract

| Direction | Python type | Shape | Dtype | Range / encoding |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(3,)` | `float32` | `[cos(theta), sin(theta), theta_dot]` |
| **Return** | `numpy.ndarray` | `(1,)` | `float32` | torque in `[-2.0, 2.0]` |

The `obs` components decompose as:

  - `obs[0]` = `cos(theta)` — bounded `[-1, 1]`
  - `obs[1]` = `sin(theta)` — bounded `[-1, 1]`
  - `obs[2]` = `theta_dot` — angular velocity rad/s, bounded `[-8, 8]`

Angle convention: `theta = 0` is **upright** (the target), `theta = ±π`
is hanging straight down. To recover the angle:
`theta = math.atan2(obs[1], obs[0])`.

Returning a Python `list` of one float (e.g. `[u]`) also works — the
env coerces — but `numpy.ndarray(shape=(1,), dtype=float32)` is the
canonical type and matches the starter.

**Pre-clip recording**: out-of-bound actions are clipped by the env
before the dynamics step, but the value your policy returned is what
gets recorded in `trajectory.jsonl` (per SPEC §4.2). The starter
clips defensively; you may or may not.

### Single-step example

```python
import math
import numpy as np

obs = np.array([math.cos(2.5), math.sin(2.5), -0.3], dtype=np.float32)
# theta = atan2(sin, cos) = 2.5 rad  (≈ 143°, pendulum is far from upright)
# theta_dot = -0.3 rad/s  (slowly rotating clockwise)

action = policy.act(obs)
# action.shape == (1,), action.dtype == float32, -2.0 <= action[0] <= 2.0
```

## Space declarations (mirrors `GET /info:env_meta`)

```json
{
  "obs_space": {
    "type": "Box",
    "shape": [3],
    "low":  [-1.0, -1.0, -8.0],
    "high": [ 1.0,  1.0,  8.0],
    "dtype": "float32"
  },
  "action_space": {
    "type": "Box",
    "shape": [1],
    "low":  [-2.0],
    "high": [ 2.0],
    "dtype": "float32"
  }
}
```

## Reward

Per step: `-(theta² + 0.1·theta_dot² + 0.001·u²)`. Always non-positive.
Best achievable episode return on a 200-step episode is bounded above
by `0`. There are no `reward_components`.

## Episode structure

- 200 steps per episode (no natural termination — the time limit is
  the only end condition; expect `truncated=true` on the last step).
- Initial state at each `reset()` is drawn uniformly from
  `theta ∈ [-π, π]` and `theta_dot ∈ [-1, 1]` by the env's hidden seed.
- The seed-to-instance mapping is hidden; you address an env instance
  by integer ID in `[0, n_env_instances)`. Submitting to the same ID
  twice gives you the same trajectory under a deterministic policy.

