# EvoPolicyGym Molecules Benchmark

This independently installable distribution implements the public rules of
[AtCoder Heuristic Contest 057 A, Molecules][task]. It depends only on the
public EvoPolicyGym SDK.

Each Episode contains 300 moving points on a 100,000 × 100,000 torus and lasts
1,000 turns. Before each simultaneous movement phase, the Policy may bond any
number of pairs from different connected components. Bonding costs rounded
toroidal distance and combines component velocity through momentum
conservation. A completed solution must contain exactly ten components of
exactly 30 points.

## Independent implementation

The package uses the public task specification as the authority for generation,
bonding, double-precision movement, terminal constraints, and scoring. Its
simulator, deterministic generator, tests, and baseline were written
independently for EvoPolicyGym. It contains no AtCoder statement text, source
code, generated inputs, seeds, submissions, visualizer, or other assets. The
official tool archive is not a dependency or redistributed component.

## Policy interface

Every observation provides the current point positions, component velocities,
and canonical component labels. The first observation also provides the fixed
space and target parameters. One Action has exactly this shape:

```python
{"bonds": [[point_i, point_j], ...]}
```

An empty list advances time without bonding. Each complete bond set is
validated before state changes; cycles, repeated same-component bonds,
out-of-range points, and impossible final partitions are rejected.

The primary metric is mean official logarithmic cost score and higher is
better. A Policy failure contributes zero. Feedback publishes aggregate
outcomes and a bounded semantic trace containing the initial public point set
and selected bond events. Because the upstream reward is zero until the final
turn, every step also reports bond count and per-bond cost, cumulative and
maximum costs, remaining required merges, merge-completion fraction, component
size histogram, singleton and completed-size component counts, whether the
mandatory 10-by-30 partition is already ready, and the score upper bound if no
additional cost were incurred. Feedback aggregates bonding frequency, cost per
bond, empty-turn fraction, and the first turn on which a valid final partition
was formed.

The packaged baseline bonds consecutive blocks of 30 points on turn zero. It
is intentionally weak but completes the full official task for every Case.

## Development

From the repository root:

```console
cd environments/atcoder/ahc057/molecules
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation tests is explicitly unsafe and
provides no isolation. The test Program is a trusted package fixture.

[task]: https://atcoder.jp/contests/ahc057/tasks/ahc057_a
