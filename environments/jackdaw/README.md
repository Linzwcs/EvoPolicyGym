# Jackdaw collection

This collection contains independently distributed game Benchmarks backed by
the Jackdaw headless rules engine.

| Environment | Package | Engine ownership |
| --- | --- | --- |
| [Balatro](balatro/) | `evopolicygym-benchmark-balatro` | Pinned and documented under `balatro/vendor/jackdaw/` |

The collection itself is not installable. Balatro owns its `pyproject.toml`,
lockfile, vendored engine revision, semantic observation adapter, replay,
baseline Program, optional authoring skill, and tests.
