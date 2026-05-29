# Hopper-v5 — MuJoCo Locomotion

## Goal

Hop forward on a single leg without falling.

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(11,)` | `float64` | unbounded (centered) |
| **Return** | `numpy.ndarray` | `(3,)` | `float32` | `[-1.0, 1.0]` per joint |

Observation components: torso angle + 3 joint angles + corresponding velocities + height (11-D).

## Reward

+1 alive bonus per step + forward velocity - control cost. Episode terminates on fall.

## Episode structure

- Up to 1000 steps. Terminates early on agent fall (where
  applicable).
- Initial state at each `reset()` includes small Gaussian
  perturbations on joint angles + velocities (Gymnasium default).

## Strategies you may take

Single-leg balance is unstable; periodic hop cycle (compress -> extend -> air) needed. Random gets ~5 (falls in 5-10 steps).

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

See `agents/pd_pendulum/policy.py` for a working reference Policy on
a different env (illustrating the contract).
