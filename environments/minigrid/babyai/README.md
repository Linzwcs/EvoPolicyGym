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
Every Episode is procedurally generated from its private seed. Composite
profiles can choose an actual task horizon below their advertised profile
maximum; per-step Feedback reports the exact remaining budget. The selected
non-debug profiles end naturally only after verified success. Wrong and
out-of-order actions normally continue, so traces distinguish successful and
failed pickup/drop/toggle attempts, door and box state changes, carried/front
objects, discovery, blocked movement, observation novelty, and timeouts.
Long traces retain the first 128 and final 32 transitions; Feedback and each
Episode trace row explicitly report the retained and omitted coverage. Each
traced observation also lists visible objects with their color and state.

The packaged initial Program is a public-observation exploration baseline. It
opens ordinary doors and clears movable blockers, but deliberately does not
encode BabyAI's full instruction grammar; the Benchmark is intended for Agent
policy development.

Tests using `ProcessExecution.unsafe()` execute trusted packaged code only.
That process backend is not a sandbox.
