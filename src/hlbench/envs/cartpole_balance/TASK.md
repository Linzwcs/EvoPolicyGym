# CartPole-Balance — Easy Anchor for the v1 Suite

## Goal

Balance a pole hinged on a cart by applying discrete left or right
force per step. Episode terminates on:
- Pole angle exceeds ±12° (the pole falls).
- Cart position exceeds ±2.4 (off the track).
- 500 steps elapsed (max episode length).

Reward: **+1 per step alive**. Max episode return = 500. "Solved" per
Gymnasium docs: mean return ≥ +475 over 100 episodes.

## Why this is in the v1 suite

This is the **EASY anchor** of the suite. Even a hand-coded angle-based
PD controller (push opposite to the pole's lean) reaches near the 500
ceiling immediately. Having one such env in the v1 paper table gives
the score histogram a clean baseline point.

## Spaces

| Direction | Type | Shape | Dtype | Notes |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(4,)` | `float32` | `[cart_x, cart_v, pole_angle, pole_ω]` |
| **Return** | `int` | scalar | python int | `0` = push left, `1` = push right |

### Observation components

| Index | Component | Range |
|---|---|---|
| 0 | `cart_position` | `[-4.8, 4.8]` (terminates at ±2.4) |
| 1 | `cart_velocity` | unbounded |
| 2 | `pole_angle` (radians) | `[-0.42, 0.42]` (terminates at ±0.21 ≈ 12°) |
| 3 | `pole_angular_velocity` | unbounded |

## Strategies you may take

1. **Reactive sign-based**: `action = 0 if pole_angle < 0 else 1`.
   ~50-100 score (over-corrects, eventually fails).
2. **PD on angle**: combine `pole_angle` and `pole_angular_velocity`
   to anticipate fall. Easily reaches 500 (ceiling).
3. **PD on (angle, cart_x)**: above + small bias toward returning the
   cart to center. Robust across all 256 random initial conditions.

## Reward

`+1` per step alive. Max return = 500 (the time limit). Episode
return distribution under expert PD is `[500, 500, 500, ...]` (very
low variance).

## Episode structure

- Up to 500 steps per episode.
- Initial state at each `reset()` is drawn from a small uniform
  range around the upright equilibrium (Gymnasium default).

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> int: ...
```

This is the easiest env in the v1 suite — use it to verify the
protocol works end-to-end, then iterate on the harder envs.
