# Pendulum-from-Pixels — Visual Swing-Up

## Goal

Same physical task as Pendulum-v1: swing the rod to the upright
position (angle = 0) and stabilize. **But the agent observes a
64×64×3 RGB rendering of the pendulum instead of the state vector
`[cos, sin, dot]`.**

The policy must extract physics state from pixels:
- **Angle** can be inferred from the orientation of the dark rod in
  the rendered image (one frame is sufficient).
- **Angular velocity** requires at least a 2-frame history (current
  frame minus previous frame), or running internal state across
  `act()` calls.

## What's different from `pendulum`

| Aspect | `pendulum` | `pendulum_from_pixels` |
|---|---|---|
| Observation | `[cos(θ), sin(θ), θ̇]` Box(3) float32 | 64×64×3 uint8 image |
| Inline / external | `inline` | **`external`** |
| Obs delivery | `trajectory.jsonl["obs"]` | `observations.npy` side-car |
| Velocity available | directly (obs[2]) | must be inferred |

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(64, 64, 3)` | `uint8` | RGB pixels |
| **Return** | `numpy.ndarray` | `(1,)` | `float32` | torque in `[-2.0, 2.0]` |

### Where the observation lives

External obs storage (SPEC §4.6):
- `trajectory.jsonl` carries `"obs": null` per step.
- Per-step pixels go to `feedback/.../ep_<XXX>/observations.npy` with
  shape `(episode_length, 64, 64, 3)` dtype `uint8`.

## Reward

Identical to `pendulum`: `-(theta² + 0.1·theta_dot² + 0.001·u²)` per
step. Episode return is non-positive; closer to 0 means longer time
spent upright with low control effort.

## Episode structure

- 200 steps per episode.
- Initial state determined by env's hidden seed (same convention as
  `pendulum`).

## Strategies you may take

1. **Single-frame angle extraction**: the pendulum rod is the
   darkest line in the image. Find its orientation via image moments
   or PCA on dark pixels. Combine with a 1-frame backbone (no
   velocity) → very poor stabilization but works for swing-up.
2. **2-frame velocity estimation**: cache the previous frame in the
   `Policy` instance (instance attributes persist across steps within
   a submit). Compute angle change → estimated velocity → standard
   PD control on (angle, velocity).
3. **Optical flow proxy**: instead of explicit angle extraction,
   compute average per-pixel motion between frames; use that as a
   velocity proxy.
4. **CNN from scratch**: train a tiny CNN (within `system/` budget) on
   in-loop trajectories. Per the no-pretrained-weights rule, the
   weights must be trained from scratch using only data the policy
   collects. Likely too sample-inefficient to beat (2).

## Implementation note

The render is downsampled from Pendulum's native 500×500 render via
block-average to 64×64. This loses some precision but keeps the obs
manageable. The dark rod is still clearly visible at this resolution.
