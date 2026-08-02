---
name: optimize-crafter-policy
description: Improve an EvoPolicyGym Policy Program for the canonical RGB Crafter achievement Benchmark.
---

# Optimize a Crafter Policy

Maximize the official Crafter score across deterministic but hidden procedural
worlds. The Policy receives only a `TensorValue` containing a `64 x 64 x 3`
uint8 RGB frame. It never receives the Environment seed, global semantic map,
player position, achievement counters, reward, or privileged inventory data.

## Respect the ABI

- Export `make_policy(context)` from `policy.py`.
- Return an object with `act(observation)`.
- Decode `TensorValue.data` as row-major RGB bytes with shape `(64, 64, 3)`.
- Return an exact integer from 0 through 16. Booleans, floats, containers, and
  out-of-range integers are invalid.
- A fresh Policy instance is created for every Episode. Keep only
  Episode-local memory between calls to `act()`.
- Read only public static configuration from
  `context.environment_parameters`. Never infer or search for hidden seeds.

## Actions

```text
0  noop                 9  place_furnace
1  move_left           10  place_plant
2  move_right          11  make_wood_pickaxe
3  move_up             12  make_stone_pickaxe
4  move_down           13  make_iron_pickaxe
5  do                  14  make_wood_sword
6  sleep               15  make_stone_sword
7  place_stone         16  make_iron_sword
8  place_table
```

`do` collects or drinks from the facing tile and attacks adjacent creatures.
Placement and crafting Actions work only when their public in-game
prerequisites are satisfied.

## Achievement progression

The 22 achievements cover collection, survival, combat, placement, and
crafting. Build capabilities in a reusable dependency order:

```text
wood -> table -> wood pickaxe -> stone
stone -> stone pickaxe + furnace
stone pickaxe -> coal + iron
wood + coal + iron + nearby table/furnace -> iron pickaxe
iron pickaxe -> diamond
```

Swords improve combat survival. Water, cows, plants, sleeping, and daylight
management protect health, food, drink, and energy. The RGB frame includes the
local world view and inventory display; build explicit visual parsing and
Episode memory rather than hard-coded seed routes.

## Build a verifiable resource-facility-craft state machine

Treat progression as confirmed state transitions, not as a timer that blindly
cycles through recipe Actions. Keep an Episode-local state machine with at
least these concepts:

- **Resource belief:** visually estimate inventory counts, but distinguish an
  unconfirmed estimate from a resource whose collection was confirmed by the
  next RGB frame or HUD change.
- **Facility state:** remember whether a table or furnace placement was
  attempted, visually confirmed, and still believed to be adjacent. Moving
  away must invalidate adjacency rather than leaving a permanent
  `table_near=True` or `furnace_near=True` flag.
- **Progress stage:** advance only after observing evidence for the prior
  dependency. A useful order is survival stabilization, wood reserve, table,
  wood tool, stone reserve, stone tool, furnace, coal and iron, iron tool, and
  diamond.
- **Recovery transition:** if expected evidence does not appear after an
  Action, retry from a legal facing tile, reacquire the missing resource, or
  rebuild the facility. Do not mark an achievement complete merely because
  its Action was returned.
- **Survival interrupt:** drinking, food, sleep, combat, and escape can
  temporarily preempt crafting without erasing the current progression stage.

Make the state machine auditable through behavior. For each revision, inspect
the achievement event order in the per-Episode `trajectory.jsonl.gz` Artifacts
and use `artifact-manifest.json` to locate the lossless observation chunks.
Verify dependencies such as `collect_wood` before `place_table`, and
`place_table` before `make_wood_pickaxe`. Re-evaluate a promising unchanged
Program across additional Episodes before trusting a small batch score.

Avoid policies whose main behavior is periodic `do`, periodic sleep, or a
fixed recipe loop. Those Actions are legal but do not demonstrate that the
resource and facility preconditions were perceived and maintained.

## Optimize the actual score

Each achievement's success rate is measured across all evaluated Episodes.
The scalar score is:

```text
exp(mean(log(1 + success_percent))) - 1
```

The shifted geometric mean rewards breadth. Improving a rare missing
achievement can matter more than repeating an easy achievement. A Policy
failure receives zero achievement credit for that Episode.

## Audit inherited controller bias

Treat the packaged baseline as disposable scaffolding, not as a behavioral
prior. Before extending it, identify control flow that ignores the current RGB
observation and replace it when it creates a repeated spatial or Action motif.
In particular, do not inherit:

- fixed clockwise or counterclockwise direction cycles;
- expanding-square, spiral, or rectangular patrol routes;
- a fixed number of `do` Actions after every movement regardless of the facing
  tile;
- periodic placement or crafting attempts without visible prerequisite
  evidence.

Exploration should react to visible walkability, resources, facilities,
creatures, and recent Episode-local movement evidence. Inspect lossless RGB
observations and the Action sequence after each revision. Repeated local loops
or a dominant Action-frequency pattern are controller defects unless the
current observation and state explicitly justify them.

## Iterate safely

1. Submit the packaged baseline on a small batch.
2. Inspect the complete compressed trajectories, achievement success table,
   and selected lossless RGB observations.
3. Eliminate invalid Actions, exceptions, and timeouts first.
4. Add visual parsing for nearby terrain, creatures, facing direction, and
   inventory icons.
5. Add a compact Episode plan for resource dependencies, survival, and
   exploration.
6. Change one capability at a time and compare repeated batches because small
   achievement samples are noisy.

Keep submitted Policy source inside `program/`. Store non-submitted diagnostic
scripts, selected frames, and derived notes in the Agent-owned `analysis/`
directory. Do not access Host paths, runtime internals, or unavailable Crafter
`info` fields.
