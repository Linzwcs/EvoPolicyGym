# Car-Racing — Top-Down Visual Racing (downsampled)

## Goal

Drive a top-down 2D car around a procedurally generated track,
collecting reward per tile visited and avoiding off-track penalties.
Maximize total track coverage in the time limit.

## What's different from full CarRacing-v3

This v1 release ships a **downsampled** variant: the env returns
16x16x3 uint8 frames (vs. 96x96x3 in the underlying
`CarRacing-v3`). Block-average downsampling preserves color but
loses spatial precision. We ship this version because:

1. Full 96x96 obs (~80 KB per step as JSON) exceeds the 10 KB
   inline-obs cap (SPEC §4.6).
2. The `observations.npy` external-storage infrastructure that
   would lift this constraint is a separate, deferred PR.

The 16x16 view is **sufficient for color-based road-following
heuristics** (road = gray, grass = green, car visible) but not
for high-fidelity vision policies.

The full 96x96 CarRacing variant will be added as `car_racing_pixel`
in a future release once the external-storage infrastructure lands.

## Spaces

| Direction | Type | Shape | Dtype | Range |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(16, 16, 3)` | `uint8` | RGB pixels in `[0, 255]` |
| **Return** | `numpy.ndarray` | `(3,)` | `float32` | `[steering, gas, brake]` |

### Action encoding

| Index | Range | Meaning |
|---|---|---|
| 0 | `[-1, 1]` | steering: -1 = full left, 0 = straight, 1 = full right |
| 1 | `[0, 1]` | gas (forward throttle): 0 = off, 1 = full |
| 2 | `[0, 1]` | brake: 0 = off, 1 = full |

## Reward

Per Gymnasium docs:
- `+1000 / N` per new track tile visited (`N` = number of tiles in the
  track).
- `-0.1` per frame (encourages fast completion).
- Episode terminates early if the car drives far off-track (penalty
  in environment dynamics).

Score ranges (approximate):
- Random policy ≈ -100 (drives off track within a few frames).
- Naive constant-throttle ≈ +50 (eventually fishtails).
- Color-segmentation + PD on lateral error ≈ +200-400 (depending on
  tuning).
- Expert (~RL trained) on full CarRacing ≈ +900; on 16x16 downsampled
  variant ceiling is lower (~+500).

## Episode structure

- Up to 1000 steps per episode.
- Track is procedurally generated per seed.

## Strategies you may take

1. **Color segmentation**: detect "road pixels" (gray ≈ ±10 of
   `(100, 100, 100)`) vs. "grass pixels" (green). Compute road
   centroid; steer toward it.
2. **Naive throttle**: constant `gas=0.5, brake=0, steering=0`.
   Crashes on first turn but tests the contract.
3. **Reactive steering with horizon estimate**: look at top half of
   frame for upcoming road direction; steer in proportion to its
   lateral offset.
4. **State-machine**: distinguish straight / left-turn / right-turn
   patterns from frame, switch control law.

The 16x16 resolution is intentionally tight — there's enough signal
to do color segmentation but not enough to extract precise track
geometry far ahead. Iteration will discover which tracks are hardest
(sharp turns, low contrast) and what additional handling helps.

## The `Policy` interface

```python
class Policy:
    def __init__(self, obs_space: dict, action_space: dict, env_meta: dict) -> None: ...
    def reset(self, episode_index: int) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...
```

## Note on procedural generation

Each `env_instance` produces a unique track (Gymnasium's
`CarRacing-v3` uses the seed to generate the track). Train and
held-out pools are drawn from disjoint seed ranges (`[0, 1M)` vs
`[1M, 2M)`) but use the same track-generation distribution — so
generalization here means "your policy works on unseen tracks of
the same statistical class," not "your policy works on a different
distribution of tracks."
