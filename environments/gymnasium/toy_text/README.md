# Gymnasium Toy Text

Toy Text environments expose compact discrete state and action spaces while
retaining meaningful stochastic planning behavior.

| Environment | Package | Policy observation | Policy action |
| --- | --- | --- | --- |
| [FrozenLake](frozen_lake/) | `evopolicygym-benchmark-frozen-lake` | Named state, coordinates, and tile | Integer `0`, `1`, `2`, or `3` |

FrozenLake is parameterized at Benchmark construction time. Its selected
standard map and transition dynamics are published through
`BenchmarkSpec.environment_parameters` and delivered to every Policy in
`PolicyContext.environment_parameters`.
