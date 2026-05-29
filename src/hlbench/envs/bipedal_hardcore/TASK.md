# Bipedal-Hardcore — Procedural Terrain Locomotion

## Goal

Walk a 4-DOF biped forward as far as possible across procedurally
generated rough terrain that includes **stumps, step-like ladders,
and pits**. Each `env_instance` yields a unique sequence of these
obstacles.

Compared to the plain `bipedal_walker` env (gentle hills only), this
variant is significantly harder: a controller that smoothly walks
flat ground will fall on the first stump.

## What's different from BipedalWalker-v3

The Gymnasium underlying env is **`BipedalWalkerHardcore-v3`**.
Procedural terrain is controlled entirely by the env's RNG (seeded
via `reset(seed=...)`), so train and held-out seeds produce
**different obstacle sequences** without any wrapper. Generalization
to held-out comes from "you saw obstacles in different orders / at
different spacings during training."

## Strategies you may take

The benchmark does not prescribe a method. Realistic approaches for
a budget-constrained run:

1. **Open-loop CPG** (central pattern generator).  Sinusoidal joint
   commands tuned for forward locomotion. Will fall on the first
   serious obstacle but handles flat segments.
2. **Reactive sensor-based**.  Use the 10 LIDAR readings (last 10
   obs values) to detect upcoming obstacles and switch to an
   obstacle-clearing gait.
3. **Hybrid CPG + reflex layers**.  Default cyclic gait + reactive
   override when LIDAR shows an obstacle.

A pure reactive policy with no rhythmic component typically can't
generate forward momentum. A pure CPG without obstacle awareness
falls. **Combining the two is where iteration helps**: see which
obstacles are causing falls, add specific reflex responses.

## The `Policy` interface

Identical to `bipedal_walker`. See that env's TASK.md and the
existing `bipedal_walker/starter_policy.py` for reference.

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

### `act(obs)`

| Direction | Python type | Shape | Dtype | Notes |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(24,)` | `float32` | hull angle/vel, joint angles/vel, ground contact, 10 LIDAR |
| **Return** | `numpy.ndarray` | `(4,)` | `float32` | torques on hip-1, knee-1, hip-2, knee-2 in `[-1.0, 1.0]` |

### Observation layout

The 24-D obs vector (per Gymnasium docs):

| Index | Component |
|---|---|
| 0 | hull angle |
| 1 | hull angular velocity |
| 2 | x velocity |
| 3 | y velocity |
| 4 | hip 1 angle |
| 5 | hip 1 speed |
| 6 | knee 1 angle |
| 7 | knee 1 speed |
| 8 | leg 1 ground contact (binary) |
| 9 | hip 2 angle |
| 10 | hip 2 speed |
| 11 | knee 2 angle |
| 12 | knee 2 speed |
| 13 | leg 2 ground contact (binary) |
| 14–23 | 10 LIDAR distance readings (forward direction) |

The LIDAR readings (14–23) are essential for obstacle awareness in
the hardcore variant — they're in the same units as the rest of the
obs and bounded `[0, 1]` (normalized distances).

## Reward

Per step: forward progress + small ctrl penalty + large hull-contact
penalty (-100 if hull touches ground = fall). Per-step reward sign
varies: positive when moving forward, negative when stuck or on
control inputs. Total return is a long-run aggregate.

Per Gymnasium docs:
- Walking the full track without falling: ~+300
- Falling immediately: ~-100
- "Solved" threshold: +300 over 100 episodes (≈ never achieved by
  hand-crafted policies)

## Episode structure

- Up to 2000 steps per episode (longer than plain BipedalWalker).
- Episode terminates early on hull-contact (fall).
- Initial state at each `reset()` is determined by the env's hidden
  seed (procedural terrain layout + initial joint state).
