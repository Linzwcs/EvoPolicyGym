# ALE Atari Benchmark

ALE 0.12.0 registers 104 Atari v5 environments, but its current PyPI wheel
ships a directly runnable ROM only for `Tetris`. This distribution therefore
exposes exactly that redistributable profile and does not claim that proprietary
ROMs are installed.

```python
from atari_benchmarks import AtariBenchmark, AtariConfig

benchmark = AtariBenchmark(AtariConfig(game="Tetris"))
```

The profile uses RGB observations, the minimal five-action set, four-frame
action repeat, sticky actions with probability 0.25, and a public 27,000-action
horizon (108,000 emulator frames). Mean return is the primary metric.

Additional games should become profiles only when their ROM distribution and CI
availability are explicit. A local ROM directory is not accepted as Benchmark
configuration because Host paths must not become portable Benchmark identity.
