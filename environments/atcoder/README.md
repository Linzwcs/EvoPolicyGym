# AtCoder environments

These Benchmark distributions implement interactive tasks from public AtCoder
specifications. They do not package AtCoder problem statements, tools, input
files, submissions, or hidden tests.

Each integration records its authoritative task page and independently
implements only the Environment rules needed by the EvoPolicyGym Policy loop.

| Suite | Distribution | Policy interaction |
| --- | --- | --- |
| [AHC054](ahc054/) | `evopolicygym-benchmark-treants-forest` | Place Treants before each explorer turn |
| [AHC057](ahc057/) | `evopolicygym-benchmark-molecules` | Bond moving components at each time |
| [AHC058](ahc058/) | `evopolicygym-benchmark-apple-incremental-game` | Upgrade one machine or wait each turn |
