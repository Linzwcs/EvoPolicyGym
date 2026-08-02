# EvoPolicyGym Warehouseman Benchmark

This independently installable distribution implements the public rules of
[CodeChef WAREHOUS, Warehouseman (Challenge)][task]. It depends only on the
public EvoPolicyGym SDK.

The Policy receives the warehouse dimensions and complete public shipment
arrival order, then submits one complete forklift instruction string. The
Environment parses and executes that string atomically. It rejects illegal
movement, pickup, drop-off, load, and unload operations; incomplete solutions;
out-of-order delivery; non-ASCII instructions; and outputs longer than 500,000
characters.

## Independent implementation

The package uses the public problem contract as the authority for dimensions,
arrival generation, forklift dynamics, completion, output limit, and scoring.
Its simulator, version-stable generator, tests, and constructive baseline were
written independently for EvoPolicyGym. It contains no CodeChef statement
text, source code, submissions, generated inputs, hidden tests, or other
contest assets.

The official problem page and editorial are copyrighted CodeChef materials and
are not software dependencies or redistributed components. The package records
their URLs and provenance without granting them the package's MIT license.

## Policy interface

The initial observation has this shape:

```python
{
    "rows": 6,
    "columns": 8,
    "arrivals": [17, 2, 31, ...],
    "instruction_limit": 500_000,
}
```

The Action is the complete instruction string using the official
`N`, `W`, `S`, `E`, `P`, `D`, `LN`, `LW`, `LS`, `LE`, `UN`, `UW`, `US`, and
`UE` instruction vocabulary.

The primary metric is mean official normalized cost and lower is better. A
Policy failure contributes the bounded sentinel cost 1,000,000. Feedback
publishes aggregate cost and bounded semantic diagnostics; it does not publish
arrival permutations, Environment seeds, or raw instruction strings.

The packaged baseline is deliberately simple but complete across the full
published 6–20 row and column range. It stores shipments from far to near along
a serpentine path, then uses a deterministic sliding-tile construction to
retrieve them in numeric order.

## Development

From the repository root:

```console
cd environments/codechef/june18/warehouseman
uv sync --extra dev
uv run ruff check src tests
uv run mypy
uv run python -m unittest discover -s tests
uv build .
```

`ProcessExecution` used by the Evaluation tests is explicitly unsafe and
provides no isolation. The test Program is a trusted package fixture.

[task]: https://www.codechef.com/problems/WAREHOUS
