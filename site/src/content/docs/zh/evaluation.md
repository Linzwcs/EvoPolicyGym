---
locale: zh
page: evaluation
section: core
title: "Evaluation 与 Runs"
navTitle: "Evaluation 与 Runs"
description: "直接 Program Evaluation 与有界 Coding Agent Program-evolution Run。"
lead: "使用 evaluate() 评估一个不可变 Program，或使用 run() 让 Coding Agent 编写并提交多个候选。"
index: D4
order: 4
docsVersion: v0.3
status: draft
---

## 直接 Evaluation

`evaluate()` 使用一个结构化 `Benchmark` 评估一个不可变 `Program`：

```python
from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution
from evopolicygym_cartpole import CartPoleBenchmark

result = evaluate(
    Program.from_directory("policy/"),
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
print(result.episodes)
```

`EvaluationConfig` 不可变且有限。Benchmark 必须准确规划请求数量的确定性
Episodes。

## EvaluationResult

`EvaluationResult` 包含：

| 字段 | 含义 |
| --- | --- |
| `benchmark_id` | 稳定的公开 Benchmark identity。 |
| `program_digest` | 被评估 Program 的 SHA-256 identity。 |
| `feedback` | Benchmark 定义的分数、公开 content 与可选 artifacts。 |
| `episodes` | 经过净化的公开 Episode summaries。 |

Episode summary 绝不公开可信 scenario、Environment seed、Host path、credential
或私有 runtime evidence。

## Program-Evolution Run

`run()` 向一个 `CodingAgent` 提供改进初始 Program 的有限权限：

```python
from evopolicygym import Program, RunConfig, run
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym_cartpole import CartPoleBenchmark

result = run(
    Program.from_directory("policy/"),
    CartPoleBenchmark(),
    agent=Codex(model="gpt-5.5"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    config=RunConfig(
        split="train",
        max_submissions=20,
        episode_budget=1_000,
        seed=42,
    ),
)
```

Host 拥有 Agent task、workspace rules、预算、submit/finish commands、进程监督
与 publication。Provider 只把 Host task 转换为经过验证的 invocation，不重新定义
Run 语义。
默认由 Agent 自主决定每次 Submission 使用多少剩余 Episode 预算。
`max_episodes_per_submission` 默认为 `None`；只有 Host 需要额外限制时才设置
正整数上限。

## Submission 记账

一次被接受的 Submission：

1. 将当前 `workspace/program/` tree 冻结为 `Program`；
2. 预留并扣除请求的 Episode 预算；
3. 评估该不可变快照；
4. 原子保留 Program、Feedback、Episode summaries 与 artifacts；
5. 在 `workspace/feedback/` 发布独立的 Agent-visible copy；
6. 将 Submission ID 加入可选最终结果集合。

非法 Program capture 不消耗 Episode 预算。Evaluation 一旦开始，预留预算不会返还。
Policy failure 是已提交的计分结果；可信 Evaluation fault 会以
`evaluation_failed` 关闭 Run。

## 选择最终 Program

Agent 通过选择一个完全发布的 Submission 完成 Run。返回的
`RunResult.final_program` 是该 Submission 保留的脱离路径 Program，而不是 Agent
workspace 中可能继续变化的内容。

可能的 terminal reason 包括：

- `finished`
- `agent_exited`
- `budget_exhausted`
- `agent_failed`
- `evaluation_failed`

## Run records

本地 Run 保留 Programs、Feedback、artifacts、events、Agent invocation 与 logs，
以及终态 `run.json` manifest。记录在当前设计下可用于诊断与复现，但不可恢复：
v0.3 没有 durable ledger、crash recovery 或 resume protocol。

[查看 Run record 结构 →](../../runs/)

## 下一步

- [执行与安全 →](../runtime/)
- [Benchmark 编写 →](../authoring/)
- [阅读 CartPole package →](https://github.com/Linzwcs/EvoPolicyGym/tree/main/environments/cartpole)
