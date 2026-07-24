---
locale: zh
page: concepts
section: core
title: "核心概念"
navTitle: "核心概念"
description: "EvoPolicyGym 0.3 的领域模型与信任边界。"
lead: "Program 不可变、Evaluation 有界、Feedback 原子提交，并且每个 Episode 都获得全新的 Policy 生命周期。"
index: D2
order: 2
docsVersion: v0.3
status: draft
---

## 领域词汇

| 值 | 含义 |
| --- | --- |
| `Program` | 一个 Policy 源码目录的脱离路径、不可变、内容寻址快照。 |
| `Episode` | 一个可信场景、一个全新 Environment，以及一个全新 Policy 进程与实例。 |
| `Evaluation` | 在有限、确定性 Episode plan 上评估一个 Program。 |
| `Feedback` | 由 Benchmark 定义的公开投影，包含一个标量分数、有界 content 与可选 artifacts。 |
| `Submission` | Coding Agent 请求 Evaluation 后得到的一组 Program 与已提交 Feedback。 |
| `ProgramEvolutionRun` | Coding Agent 编辑 Programs、提交候选、读取 Feedback，并选择最终 Submission 的一次有界外层循环。 |
| `Experiment` | 保留给未来由多个可比较 Runs 组成的集合。 |

公共 SDK 使用 `Program`，而不是 `ProgramVersion`。Program 不保留 Host
源码路径；caller 之后修改原始目录也不会改变它。

## Evaluation 生命周期

```text
Program
  ↓
deterministic Episode plan
  ↓
fresh Environment + fresh Policy process
  ↓
unmodified Actions and trusted Steps
  ↓
sanitized Episode summaries
  ↓
Benchmark-defined Feedback
```

Policy 状态可以在同一 Episode 的多次 `act()` 调用之间保留，但绝不会进入下一个
Episode。跨 Episode 的改进只能通过外层 Coding Agent 编写新 Program 完成。

## Program-evolution 生命周期

```text
initial Program
  ↓
Coding Agent edits workspace/program/
  ↓
Submission → Evaluation → committed Feedback
  ↓
Coding Agent reads workspace/feedback/
  ↓
next Program or finish(selected submission)
```

`RunConfig` 固定 split、最大 submissions、总 Episode 预算、可选的单次
Submission Episode 上限、seeds 与 timeouts。该上限默认为 `None`，因此 Agent
通常自行分配预算，但不能扩展总权限。

## 信任边界

| 可信 Host 与 Benchmark 持有 | Policy 可以观察 |
| --- | --- |
| Episode scenario 与 Environment seed | 不含 Case identity 的 `PolicyContext` |
| Environment 状态与 transitions | 公开 Observations |
| Action 校验 | 自己的 Episode 内状态 |
| Rewards、评分与私有 metrics | 仅已提交的公开 Feedback |
| Run 预算与发布 | 不包含 Host path、credential、scorer 或 runtime evidence |

Policy 边界只传递有界 `PolicyValue`。路径、文件描述符、凭据、任意 Python object
与 pickle object graph 都不能跨越该边界。

## 故障归属

Policy 异常、timeout、protocol error 与非法 Action 会成为经过净化的 Policy
failure。非法 Action 绝不会被裁剪、修复、采样或替换。

可信 Environment、Benchmark、进程控制与 cleanup fault 会中止 Evaluation，
绝不会转化为 Policy penalty。

## Package 边界

基础 `evopolicygym` wheel 拥有可移植 Kernel。独立 Benchmark distribution
只依赖公共 SDK facade 与 `evopolicygym.authoring`，Kernel 不会导入这些
distribution。

可选 Firecracker 基础设施是独立产品，并不会使 formal 或隔离执行 profile
自动可用。

## 下一步

- [Policy ABI →](../policy/)
- [Evaluation 与 Runs →](../evaluation/)
- [Benchmark 编写 →](../authoring/)
- [执行与安全 →](../runtime/)
