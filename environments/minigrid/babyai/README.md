# BabyAI Benchmark Families

An independently installable EvoPolicyGym distribution covering the 40
requested BabyAI tasks. Profiles are grouped into five Benchmark identities:
`GoTo`, `Open`, `PickupPut`, `Unlock`, and `Composite`.

```python
from minigrid_babyai import BabyAIBenchmark, BabyAIConfig

benchmark = BabyAIBenchmark(
    BabyAIConfig(profile="BossLevel")
)
```

The Host selects the profile before a Run. The profile and family are visible
to the Agent, fixed for the Run, and included in the environment digest.
Upstream-generated natural-language missions remain part of the public
observation; Episode seeds and private identity never enter Feedback.

The packaged initial Program is a public-observation exploration baseline. It
opens ordinary doors and clears movable blockers, but deliberately does not
encode BabyAI's full instruction grammar; the Benchmark is intended for Agent
policy development.

Tests using `ProcessExecution.unsafe()` execute trusted packaged code only.
That process backend is not a sandbox.
