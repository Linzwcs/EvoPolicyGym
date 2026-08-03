# Jumanji Benchmarks

This independently installable distribution exposes 18 single-Policy tasks
from Jumanji 1.1.1 through EvoPolicyGym's public authoring SPI. One profile is
selected by the Host before a Run:

- logic: Game2048, GraphColoring, Minesweeper, two RubiksCube variants, two
  Sudoku variants, and SlidingTilePuzzle;
- packing and scheduling: BinPack, FlatPack, JobShop, Knapsack, and Tetris;
- routing and games: CVRP, Maze, Snake, TSP, and PacMan.

The adapter uses Jumanji's functional `reset(key)` / `step(state, action)` API
directly. Every Episode owns a fresh upstream environment and JAX state. The
public observation uses exact Python `bool`/`int`/`float` values for scalar
leaves and canonical `TensorValue` values for non-scalar arrays. The Benchmark
specification lists every flattened field path, Policy carrier, dtype, and
shape. Action masks are enforced before an action reaches Jumanji. Profiles,
action dimensions, and horizons are fixed public environment parameters.

Feedback is designed for environment debugging as well as scoring. It includes
bounded per-Episode action, metric, reward-event, termination, and failure
summaries. Up to four Episodes receive a `trace.jsonl`; each selected transition
explains the action using profile-specific component names, verifies it against
the decision-time mask, summarizes every named field, and reports task progress
such as filled cells, packed items, remaining nodes, distance to a target, or
board occupancy. The matching `episode-NNN/observations.npz` stores every named
initial, decision, and result observation array losslessly. Short Episodes are
complete; long Episodes retain at most 48 first/last, reward/terminal-event, and
evenly sampled transitions, with all omissions reported explicitly.
NPZ necessarily materializes Python scalars as shape-`[]` NumPy arrays; its
manifest and trace summaries therefore repeat each field's live Policy carrier
so an Agent does not mistake storage dtype for the `act()` input ABI.

Jumanji observations are structured task state rather than rendered RGB video,
so this package intentionally emits inspectable arrays and semantic state
instead of synthesizing a GIF. Action masks are described as discrete, joint
multi-discrete, or (for JobShop) per-component layouts. Most directional
profiles publish the action order `up`, `right`, `down`, `left`; PacMan
separately publishes its observed upstream order `up`, `left`, `down`, `right`,
`no_op`.

GraphColoring additionally publishes its sparse terminal reward rule and the
meaning of its adjacency matrix, color vector, current-node scalar, and legal
color mask. A Policy author can therefore optimize the number of colors from
the initial specification instead of spending Episodes to infer whether a
valid but one-color-per-node solution is desirable.

Minesweeper publishes its hidden-cell and adjacent-mine encoding, dense safe-cell
reward, termination conditions, mine-count scalar, step counter, and joint
row-column legality mask. The public observation never reveals mine locations;
Policies must infer risk from previously revealed clue cells.

SlidingTilePuzzle publishes the exact `[1, 2, ..., 24, 0]` goal, identifies `0`
as the empty tile, states that directional actions move that tile, and explains
the 200-move solvable reset walk and incremental correctly-placed-tile reward.

Every observation field publishes an exact `policy_path`. Dotted field-table
names are flattened schema and NPZ names, not live dictionary keys. For example,
BinPack's `ems.x1` leaf is read as `observation["ems"]["x1"]`. BinPack also
describes normalized EMS bounds, fixed item dimensions, its joint feasibility
mask, placement rule, and volume-utilization return.

FlatPack publishes its footprint encoding, clockwise rotation convention,
top-left placement coordinates, joint legality mask, and cell-coverage reward.
Jumanji can return a nonterminal FlatPack state whose mask contains no legal
action; the strict adapter treats any such masked dead end as a natural terminal
state and reports `metrics.no_legal_actions=true`, rather than requiring a
Policy to submit an action already declared invalid.

Knapsack publishes its fixed 12.5 capacity, item sampling range, feasibility
rule, dense item-value reward, and total-value objective. Trace progress includes
the packed weight and value, total and remaining capacity, and number of legal
items, so an Agent never has to infer the capacity from terminal trajectories.

Tetris publishes automatic falling and line removal, its clockwise rotations,
4x4-window column coordinates, seven-piece sampling, and exact line-clear reward
table. The adapter corrects Jumanji 1.1.1's constant-zero observation
`step_count` from the authoritative environment state. Feedback reports lines
cleared plus column heights, aggregate and maximum height, holes, and surface
bumpiness.

CVRP identifies depot 0 and customers 1-20, documents its raw capacity and
normalized demand/capacity representation, and states the exact dense-distance
objective and depot refill rule. Feedback separates unvisited customers from
depot availability and reconstructs the route, depot-return count, distance
traveled, remaining total demand, and current capacity.

TSP documents its 20 unit-square cities, zero-reward first selection, dense
inter-city distance rewards, and automatic final edge back to the first city.
Feedback reconstructs the exact partial route and reports its open-path length,
closing-edge length, current closed-tour length, and whether all cities have
been visited.

Snake names all five observation channels and the four movement directions,
including the normalized head-to-tail body encoding and the tail-vacating mask
rule. Feedback reports head, tail, fruit, ordered body path, length, fruit count,
free cells, and Manhattan distance to the current fruit.

PacMan follows its observed Jumanji 1.1.1 dynamics rather than the inaccurate
upstream action prose: actions are up/left/down/right, while the advertised
no-op is permanently masked. The specification documents the fixed maze,
counterintuitive coordinate fields, zeroed coordinate sentinels, exact pellet,
power-up, and ghost rewards, frightened timer, and three termination causes.
Feedback normalizes all positions to [row, column], separates collected pellets
and power-ups, and records the terminal reason.

Both Sudoku profiles document the internal `-1`/`0-8` encoding, zero-based
three-component action, sparse binary reward, legality rule, and dead-end
termination. The standard profile identifies its 10,000 mixed puzzles with
25-77 clues; very-easy identifies its 1,000 puzzles with 46-80 clues. Feedback
adds per-cell candidate counts, forced and zero-candidate cells, remaining legal
assignments, and an independently checked solved flag.

Rubik profiles publish face order and viewing orientation, sticker-color ids,
solved encoding, and rotations as viewed directly at the selected face. The
standard and partly-scrambled profiles declare their exact 100- and seven-move
reset scrambles, sampling with replacement, and make clear that scramble count
is not minimum solution distance. Feedback reports per-face correct stickers,
color inventory, uniform faces, and solved state while explicitly marking
misplaced-sticker count as a heuristic rather than solution distance. Terminal
metrics distinguish `solved` from `time_limit`.

JAX is intentionally pinned to the 0.5 series tested by the Jumanji 1.1.1
release. Newer unbounded JAX releases remove APIs still used by upstream
Tetris. The distribution also declares Jumanji's omitted `requests` import-time
dependency explicitly.

Sokoban is omitted because its default generator downloads a dataset while an
environment is being created. Jumanji's multi-agent environments are omitted
until EvoPolicyGym has a first-class multi-agent lifecycle and action ABI.

From the repository root:

```console
uv sync --project environments/jumanji/jumanji --extra dev
uv run --project environments/jumanji/jumanji python -m unittest discover -s environments/jumanji/jumanji/tests
uv build environments/jumanji/jumanji
```
