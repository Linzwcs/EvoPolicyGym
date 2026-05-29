# MiniGrid-KeyCorridorS6R3-v0 — MiniGrid Partially-Observable Navigation

## Goal

Navigate a corridor of 3 rooms per side; locked door requires key from a specific room; pick up target ball.

**Mission text (constant for this env):** *"pick up the ball"*

## What's different from textbook MiniGrid

MiniGrid is a partially observable gridworld benchmark with rich
procedural variation per seed. Specific maze layout, key/object
positions, and goal location all vary by seed. The agent sees only
a 7x7 egocentric window, so map state must be tracked across steps.

## Spaces

| Direction | Type | Shape | Dtype | Notes |
|---|---|---|---|---|
| **Input** `obs` | `numpy.ndarray` | `(50,)` | `uint16` | packed: see below |
| **Return** | `int` | scalar | python int | Action in `[0, 7)` |

### Observation layout (50 ints total — packed encoding)

| Index range | Component | Encoding |
|---|---|---|
| 0..48 | packed 7x7 grid | Each cell is a single uint16: `cell = type * 100 + color * 10 + state`. Decode in policy code. |
| 49 | agent `direction` | Integer in {0, 1, 2, 3} — facing right/down/left/up. |

**Decoding the packed cells**:

```python
grid = obs[:49].reshape(7, 7)
cell_type  = grid // 100       # MiniGrid object_type id (1=empty, 2=wall, 4=door, 5=key, 6=ball, 7=box, 8=goal, 9=lava)
cell_color = (grid // 10) % 10  # MiniGrid color id (0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey)
cell_state = grid % 10           # state id (0=open for doors, 1=closed, 2=locked)
```

The 7x7 window is centered ahead of the agent (not centered on the
agent). Cells outside vision are not visible.

**Note on encoding choice**: the original MiniGrid Dict obs has the
image as a 7x7x3 uint8 array (147 ints when flattened), but most
cells are empty (type=1, color=0, state=0) — 60% of the values
were redundant noise. Packing each cell to a single uint16
preserves all information while reducing obs dimensionality by 3x,
making it easier for an LLM to process per-step.
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

- Up to 1080 steps per episode.
- Initial maze layout determined by env's hidden seed.
- Episode ends on success, timeout, or hazard contact (e.g., lava).

## Strategies you may take

Procedural maze; key location varies per seed. Must track which rooms have been searched. Highly POMDP — without memory of explored cells, policy will loop.
