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

Feedback includes sanitized per-Episode summaries and a bounded diagnostic trace
for up to four Episodes. Short Episodes retain every transition; long Episodes
retain at most 48 transitions chosen from the beginning, end, non-zero reward
events, and an even sample of the remaining timeline. Each traced transition
references its exact decision and result RGB arrays in a lossless NPZ artifact.
A PNG contact sheet provides a lightweight visual overview, and a bounded
animated GIF replays up to 24 frames with step, action, and reward labels. The
GIF always retains its initial frame, final frame, and every traced non-zero
reward event before evenly sampling the remaining timeline. Feedback reports
all omitted Episodes, steps, and replay frames explicitly and never publishes
Episode seeds or Case identity.

Additional games should become profiles only when their ROM distribution and CI
availability are explicit. A local ROM directory is not accepted as Benchmark
configuration because Host paths must not become portable Benchmark identity.
