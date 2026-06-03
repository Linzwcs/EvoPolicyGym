# CartPole

## Objective

Balance a pole hinged to a moving cart for as many steps as possible.

## Policy Interface

Implement `system/policy.py` with `class Policy`. The constructor receives
`obs_space`, `action_space`, and `env_meta` as dictionaries. `reset(episode_index)`
is called before each episode, and `act(obs)` must return one action compatible
with `action_space`.

## Observation

`obs` is `[x, x_dot, theta, theta_dot]`.

| Index | Name | Meaning |
|---|---|---|
| 0 | `x` | cart position |
| 1 | `x_dot` | cart velocity |
| 2 | `theta` | pole angle in radians |
| 3 | `theta_dot` | pole angular velocity |

The episode terminates when `abs(x) > 2.4` or `abs(theta) > 12 degrees`.

## Action

Return an integer action:

| Action | Meaning |
|---|---|
| `0` | push left |
| `1` | push right |

## Reward

Reward is `1.0` per survived step. The episode ends when the cart leaves the
track, the pole angle exceeds the limit, or the time limit is reached.

## Strategy Hints

- A sign controller on `theta` gives a quick baseline.
- A PD controller on `theta` and `theta_dot` usually improves stability.
- Add a small cart-centering term using `x` and `x_dot` for robustness.
