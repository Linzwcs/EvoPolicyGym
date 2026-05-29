# Car-Racing-Pixel — Full-Resolution Visual Racing

## Goal

Drive a top-down 2D car around a procedurally generated track, collecting
reward per tile visited. Same task as `car_racing` but with the **full
96×96×3 RGB observations** (no downsample).

## What's different from `car_racing`

| Aspect | `car_racing` | `car_racing_pixel` |
|---|---|---|
| Observation | 16×16×3 uint8 (downsampled) | **96×96×3 uint8 (full)** |
| Inline / external | `inline` | **`external`** |
| Obs delivery | `trajectory.jsonl["obs"]` | `observations.npy` side-car |
| Expert baseline | ~500 | ~900 |

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(96, 96, 3)` | `uint8` | RGB pixels in `[0, 255]` |
| **Return** | `numpy.ndarray` | `(3,)` | `float32` | `[steering, gas, brake]` |

### Where the observation lives

Because each frame is ~28 KB raw (too large for inline JSON), this env
uses **external obs storage** (SPEC §4.6):

- `trajectory.jsonl` carries `"obs": null` for every step.
- Per-step observations are written to a side-car file at
  `feedback/submit_NNN/episodes/ep_<XXX>/observations.npy` with shape
  `(episode_length, 96, 96, 3)` dtype `uint8`.

The agent (or any analysis tool) reads via numpy mmap:

```python
import numpy as np
obs = np.load("feedback/submit_005/episodes/ep_042/observations.npy",
              mmap_mode="r")
frame_100 = obs[100]            # shape (96, 96, 3) uint8
```

Loading via `mmap_mode="r"` avoids paying for the whole array up
front — single-frame access is essentially free.

### Action encoding

Same as `car_racing`:

| Index | Range | Meaning |
|---|---|---|
| 0 | `[-1, 1]` | steering |
| 1 | `[0, 1]` | gas |
| 2 | `[0, 1]` | brake |

## Reward

Per Gymnasium docs (identical to `car_racing`):
- `+1000 / N` per new track tile visited (`N` = total tiles).
- `-0.1` per frame.
- Early termination if the car drives far off-track.

Approximate score ranges:
- Random ≈ -100.
- Naive throttle ≈ +50.
- Color-segmentation + PD on full resolution ≈ +400-700.
- RL-trained expert ≈ +900.

## Episode structure

- Up to 1000 steps per episode.
- Track procedurally generated per seed.
- Same train (`[0, 1M)`) vs held-out (`[1M, 2M)`) seed split convention.

## Strategies you may take

The full-resolution view enables strategies that the 16×16 lite
variant can't:

1. **Lookahead with horizon estimation**: top portion of frame
   shows track 20-30 cells ahead; estimate upcoming curvature.
2. **Pixel-level lane tracking**: locate the road centerline by
   averaging the x-coordinate of road-colored pixels in the
   foreground region.
3. **State extraction**: estimate car heading from the orientation
   of the road centerline relative to vertical.
4. **Color-segmentation + reactive control**: same approach as the
   lite variant but with much higher spatial precision — closer to
   the gap with RL-trained policies.

## How this exercises the protocol

This is the canonical test of the `obs_storage="external"` mechanism
(SPEC §4.6). If the harness or feedback writer mishandles the
side-car, this env's data will be missing or shape-mismatched. The
`tests/test_visual_envs.py` suite checks this end-to-end via the
Sandbox subprocess path.
