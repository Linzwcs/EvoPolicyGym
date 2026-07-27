# Jackdaw collection

This collection contains independently distributed game Benchmarks backed by
the Jackdaw headless rules engine.

| Environment | Package | Engine ownership |
| --- | --- | --- |
| [Balatro](balatro/) | `evopolicygym-benchmark-balatro` | Pinned and documented under `balatro/vendor/jackdaw/` |

The collection itself is not installable. Balatro owns its `pyproject.toml`,
lockfile, vendored engine revision, semantic observation adapter, replay,
baseline Program, and tests. Compatible Coding Agent Skills are independent
Run inputs maintained under the repository's top-level `skills/` directory.
