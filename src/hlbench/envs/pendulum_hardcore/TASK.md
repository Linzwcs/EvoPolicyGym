# Pendulum-Hardcore — Domain-Randomized Swing-Up

## Goal

Same physical task as Pendulum-v1: swing a frictionless rod to the
upright position (angle = 0) and stabilize. **But the rod's mass,
length, and gravity vary per episode**, drawn from a wide range. A
fixed-gain controller that works on one configuration will fail on
others.

## What's different from Pendulum-v1

Each `env_instance` corresponds to a hidden seed; the seed
deterministically maps to a `(mass, length, gravity)` triple within
the documented ranges. **You don't see the specific values.** You
only know the ranges and that the values are constant for the
duration of one episode.

### Train pool ranges

| Parameter | Train range |
|---|---|
| `mass` (kg) | `[0.5, 2.0]` |
| `length` (m) | `[0.7, 1.5]` |
| `gravity` (m/s²) | `[8.0, 12.0]` |

### Held-out pool ranges (OOD, disjoint from train)

| Parameter | Held-out range |
|---|---|
| `mass` (kg) | `[2.0, 3.5]` (heavier) |
| `length` (m) | `[1.5, 2.2]` (longer) |
| `gravity` (m/s²) | `[4.0, 8.0]` (weaker → harder swing-up) |

The held-out range is **disjoint** from the train range on every axis
(they touch at boundaries). A policy that overfits to the train
distribution provably degrades on held-out.

## Strategies you may take

The benchmark does not prescribe a method. Three structurally
different approaches:

1. **Robust fixed gains.** Find gains that work across the full
   held-out range, accepting suboptimality at every point. Simplest
   but caps your ceiling.
2. **Gain scheduling.** Estimate `(m, l, g)` from a brief excitation
   at episode start (a few `act()` calls that perturb the system),
   then choose gains as a function of the estimate.
3. **Adaptive control.** Maintain online estimates of `m, l, g`
   throughout the episode and update controller parameters.

A naive PD with gains tuned for "nominal" parameters will get strong
in-loop scores (if you're lucky with in-loop seeds) but collapse on
held-out — this is exactly what the held-out evaluation catches.

## The `Policy` interface

Identical to Pendulum-v1. See `agents/pd_pendulum/policy.py` for a
working vanilla PD reference (which will NOT generalize to held-out
on this env).

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

### `act(obs)`

| Direction | Python type | Shape | Dtype | Range / encoding |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(3,)` | `float32` | `[cos(theta), sin(theta), theta_dot]` |
| **Return** | `numpy.ndarray` | `(1,)` | `float32` | torque in `[-2.0, 2.0]` |

Angle convention: `theta = 0` upright (target). To recover:
`theta = math.atan2(obs[1], obs[0])`.

## Reward

Per step: `-(theta² + 0.1·theta_dot² + 0.001·u²)`. Always
non-positive. Episode return depends on how quickly you reach
upright and how steady you keep it. Because mass/length affect
inertia and gravity affects the restoring torque, the *same
controller* yields different returns on different seeds.

## Episode structure

- 200 steps per episode (no natural termination — only time limit).
- Initial state at each `reset()` is drawn from
  `theta ∈ [-π, π]` and `theta_dot ∈ [-1, 1]` by the env's hidden
  seed (same convention as Pendulum-v1).
- Mass, length, and gravity are constant within an episode but
  vary across episodes (across `env_instance` IDs).

## Space declarations (mirrors `GET /info:env_meta`)

Same as Pendulum-v1:

```json
{
  "obs_space": {
    "type": "Box", "shape": [3],
    "low":  [-1.0, -1.0, -8.0],
    "high": [ 1.0,  1.0,  8.0],
    "dtype": "float32"
  },
  "action_space": {
    "type": "Box", "shape": [1],
    "low":  [-2.0],
    "high": [ 2.0],
    "dtype": "float32"
  }
}
```
