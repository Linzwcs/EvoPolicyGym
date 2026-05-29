# BipedalWalker-v3 — Walk Forward on Uneven Terrain

## Goal

A 4-joint bipedal robot must walk forward over slightly uneven
terrain without falling. Reward is roughly proportional to forward
progress, with a large penalty for the hull touching the ground
(falling).

## The `Policy` you write

A starter at `workspace/system/policy.py` is auto-staged.

### Required interface

```python
class Policy:
    def __init__(self, obs_space, action_space, env_meta) -> None: ...
    def reset(self, episode_index) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

### `act(obs)` — the per-step contract

| Direction | Python type | Shape | Dtype | Range / encoding |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(24,)` | `float32` | hull state + joint states + ground-contact flags + 10 LIDAR rays |
| **Return** | `numpy.ndarray` | `(4,)` | `float32` | joint torques in `[-1, 1]` × 4 |

`obs` decomposition (per gym source):

  - `obs[0]` — hull angle (rad)
  - `obs[1]` — hull angular velocity (rad/s)
  - `obs[2..3]` — hull linear x/y velocity
  - `obs[4..5]` — hip-1 angle, hip-1 angular velocity
  - `obs[6..7]` — knee-1 angle, knee-1 angular velocity
  - `obs[8]` — leg-1 ground contact (1.0 if touching)
  - `obs[9..10]` — hip-2 angle, hip-2 angular velocity
  - `obs[11..12]` — knee-2 angle, knee-2 angular velocity
  - `obs[13]` — leg-2 ground contact
  - `obs[14..23]` — 10 LIDAR distance readings (forward & downward)

`action` decomposition:

  - `action[0]` — hip-1 torque
  - `action[1]` — knee-1 torque
  - `action[2]` — hip-2 torque
  - `action[3]` — knee-2 torque

## Space declarations

```json
{
  "obs_space": {"type": "Box", "shape": [24], "dtype": "float32"},
  "action_space": {"type": "Box", "shape": [4],
                   "low":  [-1, -1, -1, -1],
                   "high": [ 1,  1,  1,  1],
                   "dtype": "float32"}
}
```

## Reward

Roughly: forward velocity bonus per step, small joint-torque cost,
and a large one-time penalty (-100) if the hull touches the ground.
Solved threshold per the gym docs: average return ≥ +300.

## Episode structure

- 1600 steps maximum.
- Initial state: standing upright at the start of randomly-seeded
  terrain. `truncated=true` at step 1600 if still upright;
  `terminated=true` immediately on hull-ground contact.

## Strategy hints

Random torque flops the hull into the ground in a few steps. The
classical baselines:

  - **Hand-crafted gait**: alternating sinusoidal hip/knee oscillation
    at the natural step frequency. Gets you walking but not fast.
  - **Open-loop scripted gait** with phase = step_count: simple
    function of `t` produces a passable walk on flat terrain.
  - **Trained PPO/SAC**: state-of-the-art ~+320, takes ~5M steps to
    train. Out of reach with our submit budget — but a pretrained
    architecture sketched at small scale can reach +50 to +150.
  - **Reactive controller using LIDAR**: react to upcoming bumps.
