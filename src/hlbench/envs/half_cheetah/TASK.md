# HalfCheetah-v5 — MuJoCo Locomotion

## Goal

Run forward as fast as possible without falling.

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(17,)` | `float64` | unbounded (centered) |
| **Return** | `numpy.ndarray` | `(6,)` | `float32` | `[-1.0, 1.0]` per joint |

Observation components: torso pose + joint angles + joint velocities (17-D).

## Reward

Forward velocity minus control cost (small action penalty).

## Episode structure

- Up to 1000 steps. Terminates early on agent fall (where
  applicable).
- Initial state at each `reset()` includes small Gaussian
  perturbations on joint angles + velocities (Gymnasium default).

## Strategies you may take

Coordinated periodic gait outperforms reactive control. Hand-coded sinusoidal joint commands (CPG) reach ~3000 with tuning.

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

See `agents/pd_pendulum/policy.py` for a working reference Policy on
a different env (illustrating the contract).
