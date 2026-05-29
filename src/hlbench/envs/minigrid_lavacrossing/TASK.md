# MiniGrid-LavaCrossingS11N5-v0 — MiniGrid Partially-Observable Navigation

## Goal

Cross 5 lava rivers without stepping on lava to reach the goal in an 11x11 grid.

**Mission text (constant for this env):** *"get to the green goal square"*

## What's different from textbook MiniGrid

MiniGrid is a partially observable gridworld benchmark with rich
procedural variation per seed. Specific maze layout, key/object
positions, and goal location all vary by seed. The agent sees only
a 7x7 egocentric window, so map state must be tracked across steps.

## Spaces

| Direction | Type | Shape | Dtype | Notes |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(148,)` | `uint8` | `image.flatten() + [direction]` |
| **Return** | `int` | scalar | python int | Action in `[0, 7)` |

### Observation layout

| Index range | Component | Encoding |
|---|---|---|
| 0..146 | egocentric grid `image` | 7x7x3 flattened. Channel 0 = object type, channel 1 = color, channel 2 = state. See MiniGrid docs. |
| 147 | agent `direction` | Integer in {0, 1, 2, 3} — facing right/down/left/up. |

The 7x7 window is centered ahead of the agent (not centered on the
agent). Cells outside vision are not visible.

### Action set

| ID | Action | Meaning |
|---|---|---|
| 0 | turn_left | Rotate 90° counter-clockwise in place |
| 1 | turn_right | Rotate 90° clockwise in place |
| 2 | move_forward | Move one cell forward if not blocked |
| 3 | pickup | Pick up object in current cell |
| 4 | drop | Drop carried object |
| 5 | toggle | Toggle / interact with adjacent (e.g., open door) |
| 6 | done | Declare task complete |

## Reward

MiniGrid uses **sparse positive reward on success**, discounted by
episode length. Specifically:
- Success: `1 - 0.9 * (steps_used / max_steps)`. So fast success
  gives near 1.0; slow success gives near 0.1.
- Failure (timeout or hazard): 0.

`expert_baseline ≈ 0.9` corresponds to expert agents solving
within a fraction of max_steps. Random policies almost never
succeed and score 0.

## Episode structure

- Up to 880 steps per episode.
- Initial maze layout determined by env's hidden seed.
- Episode ends on success, timeout, or hazard contact (e.g., lava).

## Strategies you may take

Direct lava contact = episode termination (lava is fatal). Crossings are at random gaps; agent must find gaps via exploration. No keys; pure navigation under hazard.
