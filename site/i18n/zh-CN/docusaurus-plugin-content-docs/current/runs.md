---
locale: zh
page: runs
section: api
title: "Coding Agent Runs"
navTitle: "Runs"
description: "让一个 Coding Agent 在固定限制内修改、评估并交接 Program 候选。"
lead: "Run 为一个 Coding Agent 提供工作区、Episode 预算和固定训练池。"
index: D6
order: 6
docsVersion: v0.3
status: draft
---

## 启动 Run

```python
from cartpole import CartPoleBenchmark
from evopolicygym import Program
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import RunConfig, run

result = run(
    Program.from_directory("my-policy/"),
    CartPoleBenchmark(),
    agent=Codex(model="gpt-5.6-luna", reasoning_effort="high"),
    execution=ProcessExecution.unsafe(),
    record_to="runs/cartpole-001",
    config=RunConfig(
        max_submissions=3,
        episode_budget=30,
        episode_pool_size=60,
        max_episodes_per_submission=10,
        seed=42,
    ),
)
```

Host 在 Agent 启动前创建 Episode 池。Agent 可以修改 `workspace/program/`、提交
候选、读取已发布 Feedback，并以已发布候选结束 Run。

## Run 限制

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `split` | `"train"` | Agent 可用的 Benchmark 数据划分。 |
| `max_submissions` | `20` | 最多接受的 Submission 数量。 |
| `episode_budget` | `1_000` | 所有 Submission 累计消耗的 Episode 编号数。 |
| `episode_pool_size` | Episode 预算 | 固定的 Run-local Episode 标识数量。 |
| `max_episodes_per_submission` | `None` | 单次 Submission 的可选上限。 |
| `seed` | `0` | 用于创建训练池的种子。 |
| `episode_timeout_seconds` | `30.0` | 每个 Episode 的超时。 |
| `agent_timeout_seconds` | `3_600.0` | Agent 进程的超时。 |

池大小和预算是两个限制。复用同一个 Episode 编号可以配对比较 Program，但会再次
消耗预算，并且仍创建全新的 Environment 和 Policy 状态。

## 提交和结束

Agent 在有效 Session 中使用：

```console
evopolicygym-session submit program --episodes "0:2,4:8"
evopolicygym-session finish submission-000002
```

该选择器展开为 `0, 1, 4, 5, 6, 7`。编号必须非空、严格递增、位于池范围内，且不
超过剩余预算。

完成的 Submission 会原子发布 Program、所选编号、Feedback、Episode 结果和
artifacts。Program 捕获失败不消耗预算；Evaluation 开始后，预留预算不再返还。

## Validation 与 Assessment

未配置 `ValidationConfig` 时，Agent 以一个已发布候选结束。配置 Validation 后，
Agent 可以交接一个有序候选列表。Agent 清理完成后，Host 在相同的私有 Validation
Episodes 上评估所有候选并选择一个 Program。

`AssessmentConfig` 只在 held-out split 上评估被选中的 Program，不改变选择结果。
Validation 与 Assessment 证据不会发布到 Agent 工作区。

## Agent Skills

Agent Skills 是显式的 Run 输入：

```python
from evopolicygym.skills import AgentSkill

skill = AgentSkill.from_directory("skills/evopolicygym")
result = run(..., skills=(skill,))
```

Skills 以只读形式复制到 `workspace/skills/`，不会传入 Policy 进程。

## 结果和记录

`RunResult` 包含终止原因、已发布 Submissions、交接的候选 ID、最终 Program，以及
可选的 Validation 和 Assessment 结果。

`record_to` 指定的目录保存工作区、不可变 Programs、Feedback、artifacts、事件、
Agent 日志和 `run.json`。`v0.3` 的 Run 不能恢复执行。

:::warning 本地进程执行

`ProcessExecution.unsafe()` 不是沙箱。Agent 和 Policy 代码都以当前操作系统用户
权限运行。

:::

## 下一步

- [Evaluation](./evaluation.md)
- [执行与安全](./runtime.md)
- [Run record 结构](/runs/)
