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
public observation contains canonical `TensorValue` leaves, and action masks
are enforced before an action reaches Jumanji. Profiles, action dimensions,
and horizons are fixed public environment parameters.

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
