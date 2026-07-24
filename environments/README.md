# EvoPolicyGym Environments

This directory contains independently installable Benchmark distributions.
Each child directory owns its Environment implementation, Episode planning,
scoring, Feedback, packaged baseline, dependencies, and tests.

Environment distributions depend only on the supported public EvoPolicyGym SDK
and authoring SPI. They are not included in the portable base wheel, the Kernel
does not depend on them, and sibling Environments must not import one another.

Current Environments:

- `cartpole/`: the Gymnasium CartPole-v1 conformance Benchmark.

Add future distributions, including the planned original poker-Roguelite
Benchmark, as new independently packaged child directories.
