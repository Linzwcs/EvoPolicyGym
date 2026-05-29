# Ant-v5 — MuJoCo Locomotion

## Goal

Locomote a 4-legged ant forward.

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(105,)` | `float64` | unbounded (centered) |
| **Return** | `numpy.ndarray` | `(8,)` | `float32` | `[-1.0, 1.0]` per joint |

Observation components: torso pose + 8 joint angles/velocities + contact forces (105-D, longest in MuJoCo suite).

## Reward

+1 alive + forward velocity - control cost - large contact penalty.

## Episode structure

- Up to 1000 steps. Terminates early on agent fall (where
  applicable).
- Initial state at each `reset()` includes small Gaussian
  perturbations on joint angles + velocities (Gymnasium default).

## Strategies you may take

4 legs makes static stability easier than Walker2d, but the 105-D obs is noisy. Trotting gait (diagonal-pair) is the textbook approach.

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

See `agents/pd_pendulum/policy.py` for a working reference Policy on
a different env (illustrating the contract).
