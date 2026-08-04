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
The public parameters explicitly document that `TensorValue` is not iterable
and show the packed little-endian `float64` decoding pattern for Policies.
Run-visible traces retain only the Policy-visible state and task one-hot; they
never include Episode seeds, Host scenarios, or other private Case identity.

The primary metric is success rate. Mean return and public reward-component
traces provide intermediate optimization feedback. Current values, per-step
changes, and Episode-best values are reported for reach/contact, grasp,
in-place progress, object-to-target distance, and dense reward. Feedback also
distinguishes current versus ever-achieved success/grasp, first achievement,
later regression, action magnitude/saturation, state motion, and per-task
success/return for MT collections. Each traced transition
contains the Policy-visible observation, Action, reward, next observation, and
public metrics. Episodes longer than 160 steps retain the first 128 and final 32
steps, with retained and omitted counts reported explicitly. ML1/ML10/ML45 are
reserved until the Kernel has an explicit Trial abstraction.

Every Episode with at least one valid transition also publishes its own
bounded 128x128 animated GIF from MetaWorld's `corner2` MuJoCo camera. The Host
captures the initial state, first result, terminal result, and an adaptive
stride of intermediate results, with at most 42 frames per Episode. Raw RGB
tensors exist only in Host-side Step metrics, are removed from JSONL traces,
and never become Policy observations. GIFs use `retention="bulk"`.

Feedback includes one video manifest for every Episode. A zero-step Policy
failure has no public post-reset artifact channel, so it is retained as an
explicit unavailable video result rather than being silently omitted. Episode
seeds and private scenario identity never cross the Policy boundary.
