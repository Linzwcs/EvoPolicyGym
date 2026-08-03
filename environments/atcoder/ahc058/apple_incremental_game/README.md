# EvoPolicyGym Apple Incremental Game Benchmark

This independently installable distribution implements the public turn rules
of [AtCoder Heuristic Contest 058 A, Apple Incremental Game][task]. It depends
only on the public EvoPolicyGym SDK.

Each Episode contains the complete published 10 machine IDs, four production
levels, and 500 turns. On every turn the Policy either strengthens one machine
or waits. The Environment then processes all machines in Level 0, 1, 2, 3
order. Unaffordable or out-of-range upgrades are rejected without repair.

## Independent implementation

The package uses the public task specification as the authority for dynamics,
input generation, horizon, and scoring. Its simulator, deterministic case
generator, tests, and baseline were written independently for EvoPolicyGym. It
contains no AtCoder statement text, source code, generated inputs, seeds,
submissions, visualizer, or other assets. The official tool archive is not a
dependency or redistributed component.

Cases follow the published log-uniform capacity and initial-cost distribution
through an EvoPolicyGym-owned version-stable random stream. Episode seeds and
split identity never cross the Policy boundary.

## Policy interface

The first observation contains capacities and initial costs together with the
current apples, machine counts, powers, turn, and remaining horizon.
Subsequent observations retain only the evolving state; same-Episode Policy
state can retain the initial configuration.

Wait for one turn:

```python
None
```

Strengthen one machine:

```python
{"upgrade": [level, machine_id]}
```

The primary metric is mean `round(100_000 * log2(final_apples))`, and higher is
better. A Policy failure contributes zero. Feedback publishes aggregate
outcomes and a bounded semantic `trace.jsonl` Artifact. Since the upstream
reward is zero until turn 500, each transition also reports purchase cost and
post-purchase balance, production and net apple growth, current production
rate, spend/production totals, the score if the Episode ended now, upgrade and
wait frequencies, per-level and per-machine investment counts, and how many
upgrades are currently affordable together with the cheapest choice. Feedback
aggregates investment, production, waiting, and final affordability across
Episodes.

## Development

From the repository root:

```console
cd environments/atcoder/ahc058/apple_incremental_game
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation tests is explicitly unsafe and
provides no isolation. The test Program is a trusted package fixture.

[task]: https://atcoder.jp/contests/ahc058/tasks/ahc058_a
