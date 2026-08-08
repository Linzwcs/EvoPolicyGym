# Crafter

This ecosystem contains independently installable EvoPolicyGym Benchmark
distributions powered by [Crafter](https://github.com/danijar/crafter).

- `crafter/`: canonical RGB achievement scoring, a canonical score plus
  `mean_survival_steps / 100` profile, a variant that additionally rewards
  repeated confirmed achievement events under a continuous survival-scaled
  limit, and a stronger variant using `mean_survival_steps / 20`. The legacy
  survival-gated long-horizon profile and additive survival-development v3
  profile remain selectable over the same Crafter world. Training Feedback
  contains complete compressed trajectories and lossless RGB observations
  under an explicit temporal-retention protocol.

The distribution is not included in the base EvoPolicyGym wheel.
