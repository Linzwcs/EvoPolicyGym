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
| `ProgramEvolutionRun` | Coding Agent 编辑 Programs、提交候选、读取 Feedback，并把已发布候选交给 Host 侧选择的一次有界外层循环。 |
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
Host fixes indexed training Episode pool
  ↓
Coding Agent edits workspace/program/
  ↓
Submission(selector) → fresh runtimes → committed Feedback
  ↓
Coding Agent reads indexed outcomes in workspace/feedback/
  ↓
next Program or finish(candidate IDs)
  ↓
Host-side final selection
```

`RunConfig` 固定 split、最大 submissions、总 Episode 预算、训练 Episode 池大小、
可选的单次 Submission Episode 上限、seed 与 timeouts，并且都在 Agent 启动前
确定。池大小与预算是两个不同的限制：池大小控制可用的 Episode identity 数量，
预算控制所有 Submissions 累计选择的编号数量。池大小默认等于总预算。

Agent 从池中选择公开的 Run-local 编号。同一编号在不同 Submission 中保持隐藏
Episode specification 与 Policy seed 不变，从而支持配对比较；但每次使用仍创建
全新的 runtime 状态并再次扣除预算。池编号是实验句柄，不是 Environment seed；
Agent 与 Policy 都无法看到底层 seed。单次上限默认为 `None`。

## 信任边界

| 可信 Host 与 Benchmark 持有 | Policy 可以观察 |
| --- | --- |
| Environment 参数选择 | Evaluation 前固定的公开 `environment_parameters` |
| Episode scenario、Environment seed 与池映射 | 不含池编号或 Case identity 的 `PolicyContext` |
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

- [Policy ABI →](./policy.md)
- [Evaluation 与 Runs →](./evaluation.md)
- [Benchmark 编写 →](./authoring.md)
- [执行与安全 →](./runtime.md)
