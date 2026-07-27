# MetaWorld Benchmark

This independent distribution exposes MetaWorld 3.1.1 multi-task benchmarks
through EvoPolicyGym's single-Policy authoring surface.

`MetaWorldConfig.profile` accepts any of the 50 canonical `*-v3` task names for
MT1, plus `mt10`, `mt50`, or `custom`. A custom profile requires a fixed tuple
of canonical task names:

```python
from metaworld_benchmarks import MetaWorldBenchmark, MetaWorldConfig

mt1 = MetaWorldBenchmark(MetaWorldConfig(profile="assembly-v3"))
mt10 = MetaWorldBenchmark(MetaWorldConfig(profile="mt10"))
custom = MetaWorldBenchmark(
    MetaWorldConfig(
        profile="custom",
        custom_tasks=("reach-v3", "push-v3", "door-open-v3"),
    )
)
```

The Host fixes the task collection for a Run and chooses the concrete task for
each Episode. MT1 observations are canonical 39-element `float64`
`TensorValue`s. MT10, MT50, and custom observations add a public one-hot task
tensor with a public index-to-task mapping in the environment parameters.
Episode task selection remains absent from Feedback and Run-visible traces.

The primary metric is success rate. Mean return and public reward-component
traces provide intermediate optimization feedback. ML1/ML10/ML45 are reserved
until the Kernel has an explicit Trial abstraction.

