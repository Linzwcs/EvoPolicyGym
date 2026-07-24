---
locale: zh
page: authoring
section: extend
title: "Benchmark 编写"
navTitle: "Benchmark 编写"
description: "用于独立分发 EvoPolicyGym Benchmark 与 Environment 的公共 authoring SPI。"
lead: "Benchmark 拥有确定性 Episode 规划、全新 Environment、严格 Action 语义、评分、Feedback 与 conformance 证据。"
index: D6
order: 6
docsVersion: v0.3
status: draft
---

## Distribution 边界

Environment 被打包为可独立安装的 Benchmark distribution。它只依赖受支持的
EvoPolicyGym 公共 SDK 与 `evopolicygym.authoring`。

基础 Kernel 不导入 Benchmark distribution，多个 distribution 之间也不相互导入。

一个 distribution 拥有：

- 上游 simulator dependency；
- 静态 `BenchmarkSpec`；
- 确定性 Episode 规划；
- 每个 Episode 一个全新 Environment；
- Action 校验与可信 Steps；
- 评分与公开 Feedback；
- baseline Programs 与测试；
- 本地 conformance fixtures。

## Benchmark protocol

外部 package 实现结构化 `Benchmark` protocol：

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

对象不需要继承 framework base class。Runtime conformance 是结构化的。

## BenchmarkSpec

`BenchmarkSpec` 是静态、公开且独立于已打开 Environment 的：

| 字段 | 要求 |
| --- | --- |
| `id` | 稳定、非空的 Benchmark identity。 |
| `description` | 公开任务说明。 |
| `observation_space` | 有界 `PolicyValue` 描述。 |
| `action_space` | 有界 `PolicyValue` 描述。 |
| `metadata` | string-keyed、与 Case 无关的公开 metadata。 |
| `max_episode_steps` | 正整数 Episode hard horizon。 |
| `primary_metric` | `Feedback.score` 表示的 metric 名称。 |
| `score_direction` | 必须是 `maximize` 或 `minimize`。 |

不要在 specification 中放入私有 scenario、Environment seed、path、credential
或 scorer object。

## Episode 规划

`episodes(split, seed, count)` 必须：

- 返回准确 `count` 个 `EpisodeSpec`；
- 对相同完整输入保持确定性；
- 让 scenario value 保持可信且对 Policy 不可见；
- 只支持有文档说明的 splits；
- 不打开 Environment。

`EpisodeSpec` 包含 unsigned 64-bit `environment_seed` 与可选的有界
`scenario`。

## Environment protocol

一个全新 Environment instance 只服务一个 Episode：

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

`step()` 接收完整、未修改的 Policy Action。非法 Action 应抛出
`InvalidAction`，不能裁剪或替换。`close()` 必须能在 evaluator 的每个退出路径上
安全执行。

## Feedback

`feedback(records)` 接收可信 `EpisodeRecord` values 并返回：

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

标量 score 由 Kernel 要求。`content` 与 artifacts 由 Benchmark 定义，但必须保持
公开、有界，并且不包含 Case identity、Environment seed、Host path、credential
或私有 execution evidence。

除非 Benchmark contract 明确规定如何从经过净化的 record 评分，否则不能把
Policy failure 改写成 Environment reward。可信 fault 绝不能成为 Policy penalty。

## Conformance

公共 checker 会把固定 Action sequence 重放两次：

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

Conformance 检查结构兼容性、确定性重放、返回的 Step values、termination ordering
与 cleanup。通过 checker 只是本地证据，不代表 formal admission 或 certification。

## Package checklist

- 独立 `pyproject.toml` 与版本。
- 只依赖受支持的 `evopolicygym` series。
- 与 Kernel 不同的公开 import package。
- Package 中包含 baseline Program。
- 确定性的 `train`、`validation` 与 `test` 规划。
- 覆盖合法、非法、失败与 cleanup 路径的单元测试。
- 本地 CLI 显式要求 unsafe-process acknowledgement。
- 不导入 Kernel private package。

当前参考实现为
[`environments/cartpole`](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/cartpole)。

## 下一步

- [环境目录 →](../../environments/)
- [Evaluation 与 Runs →](../evaluation/)
- [执行与安全 →](../runtime/)
