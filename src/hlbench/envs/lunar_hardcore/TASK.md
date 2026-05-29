# Lunar-Hardcore — Wind-Disturbed Landing

## Goal

Same physical task as LunarLanderContinuous-v3: descend the lander
between the two flags using main + side engines, touch down softly,
score landing bonuses. **But wind and atmospheric turbulence are
enabled and their strengths vary per episode.** A fixed-gain PID that
lands cleanly in still air will drift off-target or crash under
stronger wind.

## What's different from `lunar_lander_continuous`

| Aspect | `lunar_lander_continuous` | `lunar_hardcore` |
|---|---|---|
| Wind | disabled | **enabled** (`enable_wind=True`) |
| `wind_power` | n/a | randomized per seed (see ranges below) |
| `turbulence_power` | n/a | randomized per seed |
| Gravity | -10 m/s² | -10 m/s² (unchanged) |
| Terrain | random per seed | random per seed (unchanged) |

Wind applies a horizontal force whose magnitude is proportional to
`wind_power`; turbulence adds Perlin-noise-driven random shocks
whose amplitude scales with `turbulence_power`.

### Train pool ranges

| Parameter | Train range |
|---|---|
| `wind_power` | `[10.0, 15.0]` |
| `turbulence_power` | `[1.0, 1.5]` |

### Held-out pool ranges (OOD, disjoint)

| Parameter | Held-out range |
|---|---|
| `wind_power` | `[15.0, 20.0]` (stronger) |
| `turbulence_power` | `[1.5, 2.0]` (more turbulent) |

The held-out range is disjoint from the train range on both axes —
overfit-to-train policies provably degrade.

## Strategies you may take

1. **High-gain stabilization.** Pump main engine + react to lateral
   velocity. Burns fuel; may crash under strong gusts.
2. **Adaptive feedforward.** Estimate wind direction from horizontal
   velocity drift during free-fall, bias side engine to compensate.
3. **Aggressive deadband + corrective burst.** Let small drifts ride,
   correct only when out-of-tolerance. Fuel-efficient on quiet seeds.

Naive "land vertically" policies tuned without wind awareness will
typically miss the landing pad horizontally and either crash on
terrain or run out of fuel hovering.

## The `Policy` interface

Identical to `lunar_lander_continuous`. See that env's TASK.md for
the obs layout (8-D: position, velocity, angle, contact flags).

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

### `act(obs)`

| Direction | Python type | Shape | Dtype | Notes |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(8,)` | `float32` | [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact] |
| **Return** | `numpy.ndarray` | `(2,)` | `float32` | [main_engine, side_engine], each in `[-1.0, 1.0]` |

Engine action conventions:
- **main**: `-1..0` = off; `0..1` = throttle increasing
- **side**: `-1..-0.5` = left burn; `-0.5..0.5` = off; `0.5..1` =
  right burn

## Reward

Per Gymnasium docs:
- Distance from landing pad: small negative per step
- Velocity (excess): small negative per step
- Tilt angle: small negative per step
- Each leg ground-contact: +10 (one-time per contact)
- Fuel burn: small negative per main-engine activation
- Land cleanly between flags: +100 (one-time)
- Crash (hull contact): -100 (one-time, terminates)

Expected ranges with wind enabled (approximate):
- Random policy: ≈ -200 (crashes early)
- Naive no-wind PID: 0 to +50 on train, -50 to -150 on held-out
- Adaptive policy: +100 to +150 on both pools

## Episode structure

- Up to 1000 steps per episode.
- Terminates on hull-contact (crash) or successful landing.
- Initial state at each `reset()` is determined by the env's hidden
  seed (terrain layout, initial position/velocity, and the
  per-episode wind/turbulence strengths).
