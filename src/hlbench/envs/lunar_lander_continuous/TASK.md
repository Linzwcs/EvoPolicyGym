# LunarLanderContinuous-v3 — Soft Landing on the Moon

## Goal

A lander begins each episode in the upper center of the screen with a
small random push. It must descend and touch down softly between two
flags at the bottom (origin x = 0). Soft landing with both legs on the
ground earns +100; crashing earns -100; using the engines costs a
small per-step amount.

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
| **Input** `obs` | `numpy.ndarray` | `(8,)` | `float32` | position / velocity / angle / leg-contact |
| **Return** | `numpy.ndarray` | `(2,)` | `float32` | engine commands in `[-1, 1]` × 2 |

`obs` decomposition:

  - `obs[0]` — x position (origin is the landing pad center)
  - `obs[1]` — y position (0 ≈ ground, 1.5 ≈ start height)
  - `obs[2]` — x velocity
  - `obs[3]` — y velocity
  - `obs[4]` — angle (rad)
  - `obs[5]` — angular velocity
  - `obs[6]` — left leg ground contact (1.0 if touching)
  - `obs[7]` — right leg ground contact

`action` decomposition (non-trivial encoding — read carefully):

  - `action[0]` — main engine.
      - `< 0`: engine off
      - `[0, 1]`: throttle proportional to value
  - `action[1]` — side thrusters.
      - `< -0.5`: left thruster on
      - `[-0.5, 0.5]`: both side thrusters off
      - `> 0.5`: right thruster on

So `[0.0, 0.0]` = main off + side off = free-fall (gravity only).

## Space declarations

```json
{
  "obs_space": {"type": "Box", "shape": [8], "dtype": "float32"},
  "action_space": {"type": "Box", "shape": [2],
                   "low":  [-1, -1], "high": [1, 1],
                   "dtype": "float32"}
}
```

## Reward

Per-step shaping (closer to the pad / slower descent / more upright = better)
plus a one-time landing/crash bonus. Solved threshold per gym docs:
average return ≥ +200.

## Episode structure

- 1000 steps maximum.
- Terminates immediately on contact (soft landing OR crash) with the
  appropriate bonus/penalty. Truncates at 1000 steps if still aloft.
- Initial state: lander near top-center with a small random push.

## Strategy hints

A do-nothing policy (`[0, 0]`) falls under gravity and crashes — return
≈ -150. The classical approaches:

  - **Hover-then-descend PD**: throttle main engine to balance gravity
    (~0.5), use side thruster to keep angle near 0 and x near 0; throttle
    down once close to the pad. Cheap to write, gets ~+100.
  - **Bang-bang reactive controller** on vertical velocity + angle.
  - **MPC / LQR** around the hover equilibrium.
  - **Trained PPO/SAC** — also fair game; the dense shaping reward
    makes this one of the easier RL targets.
