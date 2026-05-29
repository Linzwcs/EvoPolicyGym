# Acrobot-v1 — Two-Link Swing-Up

## Goal

Swing the lower link of a 2-link pendulum upward until its tip rises
above a horizontal bar (one link-length above the upper pivot).
Starting state is the system hanging straight down with small random
perturbation; the agent applies discrete torque ±1 or 0 to the second
joint to build the necessary energy and time the swing-up.

## The `Policy` you write

A starter at `workspace/system/policy.py` is auto-staged. Edit it; the
class name `Policy` and the three method signatures MUST stay exactly
the same.

### Required interface

```python
class Policy:
    def __init__(self,
                 obs_space: dict,
                 action_space: dict,
                 env_meta: dict) -> None: ...

    def reset(self, episode_index: int) -> None: ...

    def act(self, obs: np.ndarray) -> int: ...
```

### `act(obs)` — the per-step contract

| Direction | Python type | Shape | Dtype | Range / encoding |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(6,)` | `float32` | `[cos(t1), sin(t1), cos(t2), sin(t2), w1, w2]` |
| **Return** | `int` | scalar | — | `0`, `1`, or `2` |

The 3-valued action encodes torque applied at the elbow joint:

  - `0` — torque `-1`
  - `1` — torque `0` (no torque)
  - `2` — torque `+1`

Returning a `numpy.int64` also works (env coerces), but Python `int`
is the canonical type.

### `obs` decomposition

  - `obs[0..1]` = `cos(theta1), sin(theta1)` — shoulder joint angle
  - `obs[2..3]` = `cos(theta2), sin(theta2)` — elbow joint angle
    (RELATIVE to first link)
  - `obs[4]` = `theta1_dot` — shoulder angular velocity rad/s
  - `obs[5]` = `theta2_dot` — elbow angular velocity rad/s

Recover absolute angles via `math.atan2(sin, cos)`. The "tip" position
is up-from-pivot = `-cos(theta1) - cos(theta1 + theta2)` (both link
lengths = 1).

### Single-step example

```python
import math
import numpy as np

obs = np.array([math.cos(2.5), math.sin(2.5),
                math.cos(0.1), math.sin(0.1),
                -0.3, 0.5], dtype=np.float32)
# theta1 ≈ 2.5 rad (lower link 143° below horizontal from pivot down)
# theta2 ≈ 0.1 rad (upper link nearly aligned with lower)

action = policy.act(obs)
# action in {0, 1, 2}
```

## Space declarations (mirrors `GET /info:env_meta`)

```json
{
  "obs_space": {
    "type": "Box",
    "shape": [6],
    "low":  [-1.0, -1.0, -1.0, -1.0, -12.566, -28.274],
    "high": [ 1.0,  1.0,  1.0,  1.0,  12.566,  28.274],
    "dtype": "float32"
  },
  "action_space": {"type": "Discrete", "n": 3}
}
```

## Reward

`-1` per step until terminal (tip above the bar). Episode return is
therefore in `[-500, -1]`: better policies finish sooner.

## Episode structure

- 500 steps maximum; episode ends earlier when the tip clears the bar
  (`terminated=true`) or hits the time limit (`truncated=true`).
- Initial state at `reset()` is the down-hanging equilibrium with each
  state component perturbed uniformly in `[-0.1, 0.1]`.

## Strategy hints

Naive constant torque does very little — the system has to build energy
by pumping at the natural frequency. Common approaches:

  - **Energy-based swing-up**: increase total energy until it exceeds
    the up-equilibrium energy, then switch to stabilization.
  - **LQR around the up-equilibrium** once linearization holds (close
    to upright AND low angular velocity).
  - **Discrete-action heuristic on `theta2_dot` sign** — push elbow
    in the direction it's already moving when the system is below
    horizontal, brake when above. Surprisingly effective baseline.
  - **RL trained from rollout data** (PPO / DQN) — also fair game.
