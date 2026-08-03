# Crafter Player Guide

Crafter is an open-world survival game. You are not merely exploring a map or
collecting isolated rewards: you are trying to keep one vulnerable character
alive while gathering food and water, surviving repeated day-night cycles,
building shelter, and progressing from raw materials to tools and weapons.

The world is procedurally generated, so fixed routes do not transfer between
episodes. Use the visible terrain, creatures, player status, and inventory to
decide what is possible now.

## Read the screen

The RGB observation has two parts:

- The upper portion is a local 9 by 7 tile view centered on the player. It
  shows terrain, resources, creatures, placed structures, projectiles, and the
  current daylight level.
- The lower two rows are the inventory and status display. Each item is shown
  with a count from 0 through 9.

The display order is:

```text
row 1: health, food, drink, energy, sapling, wood, stone, coal, iron
row 2: diamond, wood pickaxe, stone pickaxe, iron pickaxe,
       wood sword, stone sword, iron sword
```

The player sprite indicates the facing direction. Movement attempts always
change the facing direction, even when the destination is blocked.

## Actions

```text
0  noop                 9  place_furnace
1  move_left           10  place_plant
2  move_right          11  make_wood_pickaxe
3  move_up             12  make_stone_pickaxe
4  move_down           13  make_iron_pickaxe
5  do                   14  make_wood_sword
6  sleep                15  make_stone_sword
7  place_stone         16  make_iron_sword
8  place_table
```

Most interactions concern the tile directly in front of the player:

- `do` drinks, gathers a material, harvests a ripe plant, or attacks a
  creature on the facing tile.
- A movement Action aimed at an impassable tile turns the player toward that
  tile without entering it. A following `do` can then interact with it.
- A placement Action places on the facing tile. The tile must be empty and
  valid for that structure.
- A crafting Action does not use the facing tile, but its required utility
  blocks must be within one tile of the player.

An Action whose requirements are not satisfied simply has no useful effect.
Do not treat an attempted Action as proof that gathering, placement, or
crafting succeeded; check the next screen and inventory.

## Staying alive

Every episode starts with 9 health, 9 food, 9 drink, and 9 energy. Resource and
tool inventory starts empty. Health reaching zero ends the episode permanently;
there is no revival within an episode.

While awake:

- food decreases by one after roughly every 26 Actions;
- drink decreases by one after roughly every 21 Actions;
- energy decreases by one after roughly every 31 Actions.

When food, drink, and energy are available, health slowly regenerates. If any
necessity is missing, health eventually degenerates. Starvation and
dehydration are therefore delayed threats: waiting until health starts falling
is much too late to begin searching for supplies.

### Water and food

- Face an adjacent water tile and use `do` to gain one drink, up to the maximum
  of 9. Water remains after drinking, so a known shoreline is a reusable water
  source.
- A cow has 3 health. An unarmed attack deals 1 damage, so killing a healthy
  cow normally requires three adjacent `do` Actions. Killing it supplies 6
  food.
- A planted sapling becomes a ripe plant only after more than 300 nearby world
  updates. Harvest a ripe plant with `do` for 4 food. An unripe plant gives no
  food.

### Energy and sleep

Sleeping restores energy but does not make world time run faster and does not
instantly skip to morning. Each sleeping step advances the world by one normal
update. While sleeping, food and drink are consumed at half their awake rates,
and health recovers about twice as quickly when the other necessities are
present. Sleep ends automatically when energy is full.

A nearby zombie deals 2 health damage per normal hit but 7 damage when it hits
a sleeping player. Taking damage wakes the player. Sleeping in open terrain
does not provide protection from either creatures or projectiles.

## Day, night, and enemies

One light-dark cycle lasts 300 Actions. The first episode begins in daylight,
is brightest near step 60, becomes substantially dark after about step 150,
is darkest near step 210, and returns to daylight by step 300. This cycle then
repeats.

Darkness is not cosmetic. Zombies become much more numerous as daylight
decreases. Skeletons are common around cave paths and can fire arrows from a
distance.

- A zombie has 5 health, pursues a nearby player, and attacks from an adjacent
  tile. It takes five unarmed hits, three wood-sword hits, two stone-sword
  hits, or one iron-sword hit to defeat a healthy zombie.
- A skeleton has 3 health and shoots along cardinal directions. Solid terrain
  blocks its projectiles, while water does not.
- Swords change attack damage: unarmed 1, wood sword 2, stone sword 3, and iron
  sword 5.
- Lava kills the player immediately when entered.

Combat does not prevent additional monsters from spawning. An unarmed
character can also be hit while approaching or fighting a creature.

## Ground movement and collision

Movement depends on both the ground material and whether an object already
occupies the destination tile. The player, cows, zombies, skeletons, plants,
and other objects cannot share a tile. A movement attempt into an occupied or
non-walkable tile changes the player's facing direction but not position.

The terrain rules are:

| Tile material | Player | Cow, zombie, skeleton | Skeleton arrow |
|---|---|---|---|
| grass, path, sand | can enter if unoccupied | can enter if unoccupied | can cross |
| water | cannot enter | cannot enter | can cross |
| lava | enters but dies immediately | cannot enter | can cross |
| tree | cannot enter | cannot enter | stops |
| stone, coal, iron, diamond | cannot enter | cannot enter | stops |
| table, furnace | cannot enter | cannot enter | destroys the block on impact |

Ground creatures do not mine or remove terrain. In particular, cows, zombies,
and skeletons cannot cut trees or break natural or placed stone. A skeleton's
arrow is different from creature movement: it crosses water and lava, stops at
solid terrain, and removes a table or furnace that it hits.

The player can change some blocking tiles with `do`: trees require no tool,
stone and coal require a wood pickaxe, iron requires a stone pickaxe, and
diamond requires an iron pickaxe. Mining turns those resource tiles into
walkable ground. Monsters do not have an equivalent mining action.

`place_stone` turns the facing tile into the same stone material used by
natural stone terrain. Placed stone therefore has the same collision,
projectile-blocking, and wood-pickaxe mining rules as natural stone. These are
tile mechanics; the game does not assign a special "shelter" state to an area.

## Stone shelter and safe waiting

A shelter is created through ordinary terrain collision rather than a special
game state. An area is enclosed only when every walkable approach to the player
is blocked by solid terrain. Natural and placed stone are especially useful
boundaries because ground creatures cannot enter, mine, or remove them, and
stone also stops skeleton arrows.

A player with a wood pickaxe can mine into a sufficiently thick natural stone
region to create an interior pocket. Closing its remaining walkable opening
with placed stone can join the entrance back into the surrounding solid rock.
Suitable natural geometry can therefore reduce the amount of collected stone
needed for an enclosure.

Collected stone can instead be placed as a continuous artificial boundary
around an interior area. This works without a useful natural rock formation
but normally requires more stone. One isolated stone block, an incomplete
wall, or an open entrance is not an enclosure; creatures can walk around it.

Sleeping is a way to wait more resource-efficiently inside an already enclosed
area, not a substitute for making that area safe. The world and all creatures
continue updating normally while the player sleeps. Because sleep ends when
energy becomes full, it also does not guarantee that one sleep period lasts
until daylight.

## Gathering rules

Use `do` while facing the resource.

| Source tile | Requirement | Result |
|---|---|---|
| tree | none | 1 wood; tree becomes grass |
| grass | none | 10% chance of 1 sapling |
| water | none | 1 drink; water remains |
| stone | wood pickaxe | 1 stone; tile becomes path |
| coal | wood pickaxe | 1 coal; tile becomes path |
| iron | stone pickaxe | 1 iron; tile becomes path |
| diamond | iron pickaxe | 1 diamond; tile becomes path |

Grass, path, and sand are normally walkable. Water, trees, stone, ores, tables,
and furnaces block ordinary movement. Do not confuse `place_stone` with mining
stone: collecting stone always uses `do` and requires a wood pickaxe.

## Placement rules

Placement affects the empty tile directly in front of the player.

| Structure | Cost | Valid ground |
|---|---:|---|
| table | 2 wood | grass, sand, or path |
| furnace | 4 stone | grass, sand, or path |
| stone | 1 stone | grass, sand, path, water, or lava |
| plant | 1 sapling | grass |

Do not place a table or furnace into a creature, resource object, occupied
tile, or invalid terrain. After placing a production facility, stay within one
tile while crafting; walking away can put it out of range.

## Crafting manual

Crafting consumes inventory immediately and requires the listed nearby
facility. The correct Action numbers matter: Action 11 makes a wood pickaxe,
whereas Action 14 makes a wood sword.

| Product | Action | Ingredients | Nearby facility |
|---|---:|---|---|
| wood pickaxe | 11 | 1 wood | table |
| stone pickaxe | 12 | 1 wood + 1 stone | table |
| iron pickaxe | 13 | 1 wood + 1 coal + 1 iron | table + furnace |
| wood sword | 14 | 1 wood | table |
| stone sword | 15 | 1 wood + 1 stone | table |
| iron sword | 16 | 1 wood + 1 coal + 1 iron | table + furnace |

The main production chain is:

```text
3 wood minimum
  -> spend 2 wood to place a table
  -> spend 1 wood at the table to make a wood pickaxe
  -> mine stone and coal
  -> use wood + stone at the table to make a stone pickaxe
  -> mine iron
  -> spend 4 stone to place a furnace
  -> use wood + coal + iron near both table and furnace for iron equipment
  -> use an iron pickaxe to mine diamond
```

The minimum total cost of a table followed by one wood tool is 3 wood. A table,
wood pickaxe, and wood sword together cost 4 wood.

## Rules that are easy to misread

- Sleep advances world time and offers no protection of its own.
- One adjacent stone block does not enclose the player; monsters can walk
  around an open side.
- A cow and a zombie have multiple health points, so one successful `do` does
  not normally defeat them.
- A crafting Action without the ingredients or nearby facilities has no
  useful effect.
- Wood-pickaxe Action 11 and wood-sword Action 14 are different Actions.
- A table can become too distant for crafting after the player moves away.
- Stone collection requires `do` and a wood pickaxe; `place_stone` performs the
  opposite operation and consumes inventory stone.
- Lava is enterable by the player but sets health to zero immediately.
