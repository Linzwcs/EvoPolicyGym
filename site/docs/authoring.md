---
locale: en
page: authoring
section: extend
title: "Benchmark authoring"
navTitle: "Benchmark authoring"
description: "The public authoring SPI for independently distributed EvoPolicyGym Benchmarks and Environments."
lead: "A Benchmark owns deterministic Episode planning, fresh Environments, strict Action semantics, scoring, Feedback, and conformance evidence."
index: D8
order: 8
docsVersion: v0.3
status: draft
---

## Distribution boundary

An Environment is packaged as an independently installable Benchmark
distribution. It depends only on the supported public EvoPolicyGym SDK and
`evopolicygym.authoring`.

The base Kernel does not import Benchmark distributions, and sibling
distributions do not import one another.

A distribution also owns the Policy-facing observation contract. It may
translate an upstream simulator's positional array into a bounded semantic
dictionary before the value crosses the Policy boundary. Public traces should
retain that same value so Policy inputs and diagnostic evidence agree.

A distribution owns:

- its upstream simulator dependency;
- typed Environment parameter validation and application;
- static `BenchmarkSpec`;
- deterministic Episode planning;
- one fresh Environment per Episode;
- Action validation and trusted Steps;
- scoring and public Feedback;
- baseline Programs and tests;
- local conformance fixtures.

## Benchmark protocol

External packages implement the structural `Benchmark` protocol:

```python
from collections.abc import Sequence

from evopolicygym.authoring import (
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)


class ExampleBenchmark:
    @property
    def spec(self) -> BenchmarkSpec:
        ...

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        ...

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        ...

    def feedback(
        self,
        episodes: Sequence[EpisodeRecord],
    ) -> Feedback:
        ...
```

The object does not need to inherit from a framework base class. Runtime
conformance is structural.

## BenchmarkSpec

`BenchmarkSpec` is static, public, and independent from an open Environment:

| Field | Requirement |
| --- | --- |
| `id` | Stable, non-empty Benchmark identity. |
| `description` | Public task description. |
| `observation_space` | Bounded `PolicyValue` description. |
| `action_space` | Bounded `PolicyValue` description. |
| `metadata` | String-keyed, Case-independent public metadata. |
| `environment_parameters` | Exact public values bound to Environment construction. |
| `max_episode_steps` | Positive hard Episode horizon. |
| `primary_metric` | Name of the score represented by `Feedback.score`. |
| `score_direction` | Exactly `maximize` or `minimize`. |

Do not put private scenarios, Environment seeds, paths, credentials, or
scorer objects in the specification.

Coding Agent Skills are orthogonal to Benchmark authoring. A distribution may
document a compatible Skill, but must not load or reference it from
`BenchmarkSpec`. The Run caller explicitly captures selected Skill
directories and the Host publishes them read-only under `workspace/skills/`.

The distribution owns its typed constructor. For example,
`FrozenLakeBenchmark(map_name="8x8", is_slippery=True)` validates and retains
those arguments, publishes the exact values through `environment_parameters`,
and applies the same bound values in `make_environment()`. The Kernel computes
`environment_digest`; authors do not provide it.

## Episode planning

`episodes(split, seed, count)` must:

- return exactly `count` `EpisodeSpec` values;
- be deterministic for the same complete input;
- keep scenario values trusted and Policy-invisible;
- support only documented splits;
- avoid opening an Environment.

An `EpisodeSpec` contains an unsigned 64-bit `environment_seed` and an optional
bounded `scenario`.

## Environment protocol

One fresh Environment instance serves exactly one Episode:

```python
from evopolicygym.authoring import InvalidAction, Step
from evopolicygym.policy import PolicyValue


class ExampleEnvironment:
    def reset(self) -> PolicyValue:
        ...

    def step(self, action: PolicyValue) -> Step:
        if action not in (0, 1):
            raise InvalidAction
        ...
        return Step(
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics=private_metrics,
        )

    def close(self) -> None:
        ...
```

`step()` receives the complete unmodified Policy Action. Raise
`InvalidAction` rather than clipping or substituting it. `close()` must be safe
on every evaluator exit path.

## Feedback

`feedback(records)` receives trusted `EpisodeRecord` values and returns:

```python
Feedback(
    score=mean_return,
    content={
        "metrics": {"mean_return": mean_return},
        "summary": "Public diagnostic text.",
    },
    artifacts=(...),
)
```

The scalar score is Kernel-required. `content` and artifacts are
Benchmark-defined, but they must remain public, bounded, and free of Case
identity, Environment seeds, Host paths, credentials, and private execution
evidence.

For visual environments, preserve the original captured RGB values losslessly
in Benchmark-owned bulk artifacts with explicit step alignment and a bounded
sampling manifest. Browser video and animated GIFs are presentation
derivatives and should not be the only visual evidence. State separately
whether the artifact is complete for its configured capture schedule and
whether that schedule covers every Episode step.

Policy failure must not be rewritten as an Environment reward unless the
Benchmark contract explicitly defines such scoring from the sanitized record.
Trusted faults must never become Policy penalties.

## Conformance

The public checker replays fixed Action sequences twice:

```python
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    check_benchmark,
)

report = check_benchmark(
    benchmark,
    fixtures=(
        BenchmarkFixture(
            episode=EpisodeSpec(environment_seed=7),
            actions=(0, 1, 1),
        ),
    ),
)
report.raise_for_errors()
```

Conformance checks structural compatibility, deterministic replay, returned
Step values, termination ordering, and cleanup. Passing is local evidence, not
formal admission or certification.

## Package checklist

- Independent `pyproject.toml` and version.
- Dependency on the supported `evopolicygym` series only.
- Public import package distinct from the Kernel.
- Packaged baseline Program.
- Deterministic `train`, `validation`, and `test` planning.
- Unit tests for valid, invalid, failed, and cleanup paths.
- Explicit unsafe-process acknowledgement in any local CLI.
- If an Agent skill is supplied, package it with the distribution and keep it
  free of private Episode data.
- No private Kernel imports.

Current implemented examples are
[`environments/gymnasium/classic_control/cartpole`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/cartpole),
[`environments/gymnasium/classic_control/acrobot`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/acrobot),
[`environments/gymnasium/classic_control/mountain_car`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/mountain_car),
[`environments/gymnasium/classic_control/mountain_car_continuous`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/mountain_car_continuous),
[`environments/gymnasium/classic_control/pendulum`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/gymnasium/classic_control/pendulum),
and
[`environments/jackdaw/balatro`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/jackdaw/balatro).

## Next

- [Environment collection →](/environments/)
- [Evaluation →](./evaluation.md)
- [Coding Agent Runs →](./runs.md)
- [Execution and safety →](./runtime.md)
