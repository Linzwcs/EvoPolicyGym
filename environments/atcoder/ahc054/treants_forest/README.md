# EvoPolicyGym Treant's Forest Benchmark

This independently installable distribution implements the interactive rules
of [AtCoder Heuristic Contest 054 A, Treant's Forest][task]. It depends only on
the public EvoPolicyGym SDK.

The Policy places permanent Treants on unseen empty cells before each
adventurer move. A placement set is rejected atomically if a coordinate is
duplicate, out of bounds, already revealed, occupied, contains the flower, or
disconnects either the entrance or the adventurer from the flower. The
adventurer then reveals sight lines, selects a private target, and follows a
deterministic shortest path using the official up, down, left, right tie-break.

## Independent implementation

The package uses the public task specification as the authority for game
semantics. Its simulator, deterministic case generator, tests, and baseline
were written independently for EvoPolicyGym. It contains no AtCoder statement
text, source code, generated inputs, seeds, submissions, visualizer, or other
assets. The official downloadable tool archive does not declare a software
license and is not a dependency or redistributed component.

Cases follow the published size, obstacle-connectivity, and flower-distance
distribution using an EvoPolicyGym-owned version-stable random stream. Private
target order and Episode seeds never cross the Policy boundary.

## Policy interface

The first observation contains the forest size, entrance, flower, initial tree
coordinates, adventurer position, and the initially revealed entrance.
Subsequent observations contain the new adventurer position and cells revealed
by the preceding turn. Same-Episode Policy state can retain the initial map and
all prior placements.

One Action has exactly this shape:

```python
{"placements": [[row, column], ...]}
```

An empty list places no Treants and is always the safe baseline Action.

The score is mean adventurer movement count, capped at 2,048 turns per Episode.
Each valid turn earns one point; reaching the cap is a normal truncation and a
Policy failure contributes zero. Feedback publishes aggregate outcomes and a
bounded semantic `trace.jsonl` Artifact.

## Development

From the repository root:

```console
cd environments/atcoder/ahc054/treants_forest
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation tests is explicitly unsafe and
provides no isolation. The test Program is a trusted package fixture.

[task]: https://atcoder.jp/contests/ahc054/tasks/ahc054_a
