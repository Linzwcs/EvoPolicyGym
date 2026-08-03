---
locale: zh
page: evaluation
section: api
title: "Evaluation"
navTitle: "Evaluation"
description: "在确定性的 Episode 计划上评估一个不可变 Program。"
lead: "Program 已经确定时，使用 evaluate() 进行评估。"
index: D5
order: 5
docsVersion: v0.3
status: draft
---

## 基本用法

```python
from cartpole import CartPoleBenchmark
from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution

result = evaluate(
    Program.from_directory("my-policy/"),
    CartPoleBenchmark(),
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(
        split="validation",
        episodes=100,
        seed=42,
        episode_timeout_seconds=30,
    ),
)

print(result.feedback.score)
```

`evaluate()` 接收一个 `Program`、一个结构化 `Benchmark`、显式执行方式，以及可选的
`EvaluationConfig`。

## EvaluationConfig

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `split` | `"validation"` | Benchmark 定义的 Episode 数据划分。 |
| `episodes` | `1` | Episode 数量，必须为正整数。 |
| `seed` | `0` | 用于规划 Episode 的无符号 64 位种子。 |
| `episode_timeout_seconds` | `30.0` | 每个 Episode 的正数超时。 |

Benchmark 必须返回准确数量的 Episodes。相同 split、seed 和 count 必须得到相同计划。

## EvaluationResult

| 字段 | 含义 |
| --- | --- |
| `benchmark_id` | 公开 Benchmark 标识。 |
| `environment_digest` | 公开 Environment 参数的标识。 |
| `program_digest` | 被评估 Program 的标识。 |
| `feedback` | Benchmark 定义的分数、公开内容和 artifacts。 |
| `episodes` | 经过净化的公开 Episode 结果。 |

Episode 结果包含状态、总 reward、步数和可选的 Policy 失败代码，不包含场景、
Environment seed、Host 路径、凭据或私有指标。

## Episode 行为

每个 Episode 都使用全新的 Environment、Policy 进程、Policy 实例和临时目录。
Policy 状态只能在同一 Episode 的多次 `act()` 调用之间保留。

Policy 失败会生成经过净化的失败结果。Environment、Benchmark、执行或清理故障会
中止整个 Evaluation。

:::warning 本地进程执行

`ProcessExecution.unsafe()` 不是沙箱。Policy 以当前操作系统用户权限运行。

:::

## 下一步

- [Programs](./programs.md)
- [Policy API](./policy.md)
- [Coding Agent Runs](./runs.md)
- [执行与安全](./runtime.md)
