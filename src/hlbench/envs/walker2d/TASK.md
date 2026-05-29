# Walker2d-v5 — MuJoCo Locomotion

## Goal

Walk forward on two legs without falling.

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(17,)` | `float64` | unbounded (centered) |
| **Return** | `numpy.ndarray` | `(6,)` | `float32` | `[-1.0, 1.0]` per joint |

Observation components: torso pose + 6 joint angles + corresponding velocities (17-D).

## Reward

+1 alive per step + forward velocity - control cost. Terminates on fall.

## Episode structure

- Up to 1000 steps. Terminates early on agent fall (where
  applicable).
- Initial state at each `reset()` includes small Gaussian
  perturbations on joint angles + velocities (Gymnasium default).

## Strategies you may take

Bipedal walking requires coordinated alternating gait. Naive constant torques fall instantly. Reflex-based + periodic alternation is the simplest viable approach.

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

See `agents/pd_pendulum/policy.py` for a working reference Policy on
a different env (illustrating the contract).
