# EvoPolicyGym Environments

This directory contains independently installable Benchmark distributions.
Each child directory owns its Environment implementation, Episode planning,
scoring, Feedback, packaged baseline, dependencies, and tests.

Environment distributions depend only on the supported public EvoPolicyGym SDK
and authoring SPI. They are not included in the portable base wheel, the Kernel
does not depend on them, and sibling Environments must not import one another.

Current Environments:

- `cartpole/`: the Gymnasium CartPole-v1 conformance Benchmark.
- `balatro/`: an unofficial white-stake Red Deck Benchmark powered by the
  pinned Jackdaw headless engine.

Add future distributions as new independently packaged child directories.
