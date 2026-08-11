---
name: optimize-crafter-local-symbolic-policy
description: Improve an EvoPolicyGym Policy Program for Crafter local-symbolic-v1 observations.
---

# Optimize a local-symbolic Crafter Policy

Develop one executable Policy for hidden procedural Crafter worlds. The
observation removes RGB recognition and OCR, but it does not expose a global
map, absolute position, seeds, achievement counters, action masks, or hidden
creature and survival counters.

## Respect the ABI

- Export `make_policy(context)` from `policy.py` and return an object with
  `act(observation)`.
- Return an exact integer Action from `0` through `16`.
- A fresh Policy is created for every Episode. Keep only Episode-local memory.
- The observation is a dictionary containing exactly `terrain`, `entities`,
  `inventory`, `facing`, `sleeping`, and `daylight`.
- `terrain` and `entities` are uint8 `TensorValue` values with shape `(7, 9)`.
  Decode their bytes in row-major `[row, column]` order. The player is at
  `[3, 4]`; rows increase downward and columns increase rightward.

## Symbol tables

Terrain IDs are:

```text
0 unknown/outside  1 water  2 grass  3 stone  4 path  5 sand  6 tree
7 lava  8 coal  9 iron  10 diamond  11 table  12 furnace
```

Entity IDs are:

```text
0 none  1 player  2 cow  3 zombie  4 skeleton
5 arrow-left  6 arrow-right  7 arrow-up  8 arrow-down
9 young plant  10 ripe plant  11 fence
```

`inventory` has named exact integer counts for health, food, drink, energy,
sapling, wood, stone, coal, iron, diamond, and the three pickaxes and swords.
`facing` is `left`, `right`, `up`, or `down`; `sleeping` is bool and `daylight`
is a float in `[0, 1]`.

## Actions and evidence

The unchanged Actions are:

```text
0 noop  1 left  2 right  3 up  4 down  5 do  6 sleep
7 place_stone  8 place_table  9 place_furnace  10 place_plant
11 make_wood_pickaxe  12 make_stone_pickaxe  13 make_iron_pickaxe
14 make_wood_sword  15 make_stone_sword  16 make_iron_sword
```

Movement toward an occupied or blocked adjacent cell can change facing without
changing position. `do` affects only the facing cell. Crafting still requires
the ordinary resources and nearby facilities; no prerequisite-validity hint is
provided.

Build explicit Episode-local modules for local perception, relative world and
landmark memory, survival maintenance, exploration, production dependencies,
and combat/defense. Confirm progress from inventory and later observations
rather than assuming an attempted Action succeeded. Use local observations to
update an internally estimated position; never search for a hidden seed or
write a fixed route for particular Episode indices.

Training Feedback contains one compressed trajectory per Episode and lossless
symbolic NPZ chunks. The NPZ arrays are `terrain`, `entities`, `inventory`,
`facing`, `sleeping`, `daylight`, and `observation_indices`. Read
`artifact-manifest.json` for inventory order, facing IDs, and alignment:

```text
observation[t] -> action[t] -> observation[t + 1]
```

Inspect failure modes across unseen training Episodes. Treat repeated local
loops, stationary interaction spam, and unconfirmed crafting cycles as
controller defects unless current state supplies a reason for them.
