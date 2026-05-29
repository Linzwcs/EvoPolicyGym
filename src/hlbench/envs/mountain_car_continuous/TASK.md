# MountainCarContinuous-v0 — Underpowered Car Climbs a Hill

## Goal

A car sits in a valley at the bottom of a 1-D hilly track. It must
reach the flag on the right hill (`position ≥ 0.45`). The engine is
too weak to climb directly: the policy must oscillate the car back
and forth to build kinetic energy, then convert it into the climb.

Reward is `-0.1·u²` per step (small control penalty) plus `+100`
on the single step that reaches the flag (`terminated=true`).
A do-nothing policy gets ≈ 0; a successful aggressive-throttle policy
gets ≈ 90–96 (depending on how quickly it reaches the goal and how
much |u|² it spent).

## The `Policy` you write

A starter at `workspace/system/policy.py` is auto-staged. Edit it.

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
| **Input** `obs` | `numpy.ndarray` | `(2,)` | `float32` | `[position, velocity]` |
| **Return** | `numpy.ndarray` | `(1,)` | `float32` | engine push in `[-1.0, 1.0]` |

`obs[0]` (position) is in `[-1.2, 0.6]` — the bottom of the valley
sits around `-0.5` and the goal flag is at `0.45`.

`obs[1]` (velocity) is in `[-0.07, 0.07]`. Positive velocity = moving
right.

The action is a unitless engine push: positive accelerates right,
negative accelerates left. The dynamics also include gravity, which
pulls the car toward `position = -0.5`.

### Single-step example

```python
import numpy as np

obs = np.array([-0.4, 0.03], dtype=np.float32)
# car slightly left of valley bottom, moving right at moderate speed

action = policy.act(obs)
# action.shape == (1,), -1.0 <= action[0] <= 1.0
```

## Space declarations (mirrors `GET /info:env_meta`)

```json
{
  "obs_space": {
    "type": "Box",
    "shape": [2],
    "low":  [-1.2, -0.07],
    "high": [ 0.6,  0.07],
    "dtype": "float32"
  },
  "action_space": {
    "type": "Box",
    "shape": [1],
    "low":  [-1.0],
    "high": [ 1.0],
    "dtype": "float32"
  }
}
```

## Reward

Per step: `-0.1·u²`. On reaching the flag: `+100` once. Best
policies converge to roughly `+90 to +96` total (a small ctrl cost
amortized over the swing-up steps).

## Episode structure

- 999 steps maximum; ends earlier on success (`terminated=true`).
- Initial state at `reset()` is `position ∈ [-0.6, -0.4]` uniform,
  `velocity = 0`.

## Strategy hints

Pure positive throttle gets the car partway up the right hill, gravity
pulls it back, and so on indefinitely — without smart oscillation it
never reaches the flag and the episode times out at 999 steps.

Common approaches:

  - **Bang-bang on velocity sign**: push in the direction the car is
    already moving (`u = sign(velocity)`). Simple and surprisingly
    effective once velocity exceeds a small threshold.
  - **Energy pumping**: similar idea explicit in physics: `u = sign(velocity)`
    when total energy is below the threshold needed to reach the flag.
  - **Constant rightward push** — fails: the car oscillates near
    the valley but never builds enough energy.
  - **Trained policy** (PPO / SAC) — also fair game.
