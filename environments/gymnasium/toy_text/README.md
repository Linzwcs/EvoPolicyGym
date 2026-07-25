# Gymnasium Toy Text

Toy Text environments expose compact discrete state and action spaces while
retaining meaningful stochastic planning behavior.

| Environment | Package | Policy observation | Policy action |
| --- | --- | --- | --- |
| [Blackjack](blackjack/) | `evopolicygym-benchmark-blackjack` | Player sum, dealer card, and usable Ace | Stick or hit |
| [CliffWalking](cliff_walking/) | `evopolicygym-benchmark-cliff-walking` | State, coordinates, and terrain tile | Integer `0`, `1`, `2`, or `3` |
| [FrozenLake](frozen_lake/) | `evopolicygym-benchmark-frozen-lake` | Named state, coordinates, and tile | Integer `0`, `1`, `2`, or `3` |
| [Taxi](taxi/) | `evopolicygym-benchmark-taxi` | Taxi position, passenger, destination, and legal Actions | Integer `0` through `5` |

All four standard Gymnasium Toy Text environments are implemented as
independently installable distributions. Their selected rules or transition
dynamics are published through
`BenchmarkSpec.environment_parameters` and delivered to every Policy in
`PolicyContext.environment_parameters`. Taxi-v4 preserves Gymnasium's public
action mask as `observation.legal_actions`.
