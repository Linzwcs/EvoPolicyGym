# Crafter

This ecosystem contains the independently installable EvoPolicyGym Benchmark
distribution powered by [Crafter](https://github.com/danijar/crafter).

- `crafter/` publishes two reward profiles over the pinned Crafter 1.8.3
  simulator: the default long-horizon `lhs` objective
  and the native `canonical` achievement objective.
- Both profiles support the original `64 x 64 x 3` RGB observation and the
  explicitly selected, separately identified `local-symbolic-v1` observation.
- Training Feedback can contain complete compressed trajectories and lossless
  Policy observations. Validation and held-out test Feedback remains
  aggregate-only.

The distribution is not included in the base EvoPolicyGym wheel. See
[`crafter/README.md`](crafter/README.md) for its complete contract and local
launcher.
