---
locale: zh
page: designing-evopolicygym
title: "EvoPolicyGym：让 Coding Agent 从环境反馈中构建策略系统"
description: "为什么 EvoPolicyGym 使用交互式 Environment 与可执行 Program 研究和训练 Coding Agent。"
lead: "Coding Agent 研究 Environment，从有界 Feedback 中学习，并将结论落实为可执行策略系统。"
publishedAt: "2026-07-27"
date: "2026-07-27"
authors: [evopolicygym]
tags:
  - Design
  - Motivation
  - Architecture
status: published
---

## Coding Agent 作为策略系统专家

EvoPolicyGym 直接受到了 Jiayi Weng 的
[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/#zh)
启发。文章提出 *Heuristic Learning*：Coding Agent 吸收 reward、failure、test、
log 与 replay，再通过直接编辑软件系统来改进 programmatic policy。它让我们看到，
Agentic coding 本身可以成为一种学习过程，并且持续演化的状态能够明确保存在代码中。

EvoPolicyGym 从这个洞见出发，进一步思考如何让这一过程在不同交互式 Environment
中形成有界、可复现、可比较的评估方式。

<!-- truncate -->

在 EvoPolicyGym 的 Run 中，Coding Agent 扮演策略系统专家。它研究 Environment
及其交互接口，检查初始 Program，对有效行为形成判断，再把这些判断写成一个完整、
可执行的 Policy 系统。

这个系统可以组合领域知识、状态估计、规则、规划、搜索、记忆、算法或调优后的
参数。随着证据积累，Coding Agent 可以自由调整内部设计。它的责任，是将自己从
Environment 中学到的内容落实为能够独立决策的源代码。

编写与执行是两个清晰阶段。Evaluation 运行时，提交的 Policy 独立接收
observations 并产生 Actions。最终得到的是一个可以冻结、检查、重跑和比较的独立
策略系统。

Environment Feedback 构成了策略工程闭环：

```text
研究 Environment
    ↓
编写可执行 Policy 系统
    ↓
提交并进行 Evaluation
    ↓
分析 score、trace 与 artifacts
    ↓
诊断、重新设计并再次提交
```

这就是 EvoPolicyGym 中 Autonomous Policy Evolution 的出发点：Coding Agent
贡献专家知识与软件工程能力，Environment 提供经验性证据，持续演化的 Program
记录最终形成的策略。核心问题是，Agent 能否把有限的 Environment Feedback
有效转化为更好的可执行决策系统。

## 将专家与 Policy 分开

在 EvoPolicyGym 中，Coding Agent 与 Policy 承担不同角色。

Coding Agent 是外层策略工程师与优化器。它阅读任务说明和公开 Feedback，编辑
`workspace/program/`，并决定何时提交下一个候选。Policy 是内层决策系统。它
通过一个很小的 ABI 接收 observation，并在 Episode 运行时返回 Action。

Policy 可以在同一个 Episode 的多次 `act()` 调用之间保存状态。每个新 Episode
都会获得全新的进程和 Policy 实例，跨 Episode 的改进则由新的 Program 表达。

EvoPolicyGym 将学习放在 Episodes 之间的 Program 层。Episode-local state 支持
时序行为，Program revision 记录持续改进。每次变化都有可见的源码快照，也能与
对应的 Evaluation 证据建立清晰关系。

## 让产物成为一等对象

Workspace 支持持续编写。每个被接受的 Submission 都会把当前源码树捕获为一个
不可变、内容寻址的 `Program`。Evaluation、Feedback 与 artifacts 都属于这个精确
快照，最终结果返回从已提交候选中选出的、由 Host 保留的 Program。

这样，一个 Run 就可以被理解为一系列明确的产物：

| 对象 | 职责 |
| --- | --- |
| `Program` | 被评估的可执行 Policy 源码 |
| `Submission` | 一个不可变 Program、显式训练编号 selector 及其已提交 Feedback |
| `Run` | 有界的提交序列与最终交接 |
| `Validation` | Host 在候选之间进行选择 |
| `Assessment` | 在 held-out 数据上测量所选 Program |

Program 是持久结果，Agent transcript 与进程日志提供辅助诊断信息。

## 让 Benchmark 定义有用的 Feedback

不同 Environment 需要不同类型的证据。控制任务可能需要状态轨迹和终止原因；
卡牌游戏可能需要回合摘要、经济决策或紧凑 replay。

EvoPolicyGym 统一 Feedback 载体，每个 Benchmark 定义有用的领域内容。Feedback
始终包含一个标量 score，Benchmark 可以再加入有界的公开 values 与 artifacts。
Benchmark 同时拥有 Episode 规划、Environment 构造、Action 验证与评分语义。

Kernel 只拥有跨 Benchmark 必须一致的部分：预算、不可变 Submission、生命周期
顺序、发布、选择、记录与 Policy ABI。Environment packages 可以独立安装，并且
只依赖公开 authoring interface。

这种职责划分使项目能够像 Gym 一样形成环境生态，同时让 Kernel 保持稳定且
domain-independent。

## 将证据访问视为实验条件

Run 的 Submission 上限、总 Episode 预算、固定训练池大小与可选单次上限共同定义
实验条件。例如，16 次提交、48 个 Episode 单位与 96 个 Episode identity 的池，
表示 Agent 可以从更宽的集合中选择并观察总计 48 次；更大的池不会增加交互预算。

Host 会在 Agent 启动前构建这个编号池。每次 Submission 都指定一组非空、公开的
Run-local 编号。复用编号会保持其隐藏 Episode specification 与 Policy seed 不变，
因此两个不可变 Program 可以在配对证据上比较；但每次使用仍创建全新的
Environment 与 Policy runtime，并再次扣除预算。真实 seeds、scenarios 与池构建
过程仍由 Host 持有。

Evaluation 一旦开始，预留的 Episode 配额就会被消耗。Policy failure 与无效
Action 会作为观察到的行为被报告，从而保持提交 Program 的精确语义。Feedback
会把每个经过净化的 Episode 结果映射回公开编号，而不同 selector 之间的比较仍是
未配对证据。

Agent 利用公开的搜索 Feedback 决定下一次尝试什么。Agent 结束后，控制权回到
Host：私有 Validation 在交接的候选之间进行选择，held-out Assessment 测量最终
选中的 Program。优化 Feedback 在 `finish` 时关闭，选择和最终测量保留在 Host 侧。

将搜索、选择与最终测量分开，可以让报告结果更容易解释。

## 保持 Kernel 聚焦

EvoPolicyGym 是一套职责聚焦的基础设施。

- Agent integration 只把 Host 拥有的任务翻译为 provider invocation。
- Benchmark distribution 拥有领域语义、依赖、baseline、Feedback 与测试。
- Kernel 拥有共享的 Evaluation 与 Program evolution 生命周期。
- Policy 边界传输有界的公开 values。

目前 Codex 是第一个受支持的 Coding Agent integration，本地进程是当前 active
backend。核心 contracts 保持 provider-independent 与 backend-independent，
让未来集成继续沿用 Program、Submission、Evaluation 与 Run 的语义。

## EvoPolicyGym 支持什么

EvoPolicyGym 将可扩展的交互式 Environment、Coding Agent、版本化 Program 与
可验证的 Benchmark 证据连接起来。Agent 是研究与训练对象；Program 是它留下的
可执行证据。

### 研究能够演化策略系统的 Agent

当 Environment、初始 Program、交互预算、Feedback 可见性与选择规则保持一致时，
重复 Runs 可以研究：

- **Agent capability：**相同条件下，哪个 Coding Agent 能编写出更好的 Policy
  系统？
- **Improvement efficiency：**Agent 使用多少 Environment 交互，才能产生稳定的
  Program 改进？
- **Feedback value：**哪些 trace、diagnostics、replay 或聚合信息最能促进有效修改？
- **Evolution dynamics：**Program 的结构如何随 Submission 演化，哪些修改带来
  持久收益？
- **Selection validity：**Validation 选出的候选能否在 held-out Assessment 中
  保持优势？
- **Policy-system design：**不同 Agent 会将怎样的状态表示、规则、规划、记忆与
  控制结构写入 Program？
- **Scaling and generalization：**预算、profiles、seeds、任务复杂度与 Environment
  families 变化时，这些结果如何变化？

最终 score 测量 Agent 选择的产物，连续的不可变 Programs、Feedback、artifacts
与结果则解释 Agent 如何到达这个结果。

### 使用交互式 Environment 训练 Coding Agent

Environment 与 Benchmark 共同构成任务生成器、证据生成器与 verifier。Profiles、
scenarios 和 seeds 产生任务变化，Program Evaluation、公开 Feedback、diagnostics
与 held-out 结果提供训练信号。

同一套 Environment 生态可以支持：

- 使用 Program Evaluation 和 held-out 表现作为可验证结果的 Coding Agent RL
  与 RLVR；
- 从成功的长程 Agent trajectories 进行 SFT；
- 从可观察的 evolution records——任务上下文、公开 Feedback、代码修改、
  Submissions 与结果——进行 Agent distillation；
- 根据最终产物和结果，对高质量 Agent trajectories 进行 rejection sampling；
- 跨任务 profiles 与难度进行 curriculum learning；
- 使用中间 failure、revision 与 Evaluation 进行 process supervision。

一条 Agent evolution trajectory 覆盖完整任务：理解 Environment、编写 Program、
读取 Feedback、诊断行为、修改策略系统、提交候选并完成最终交接。这类长程记录可以
为 Coding Agent 与策略工程 Agent 提供训练材料。

```text
Environment + Benchmark
        │
        ▼
Agent 编写并持续修改 Program
        │
        ├── evolution trajectory ─────▶ Agent SFT / distillation
        └── evaluation outcomes ──────▶ Agent RL / RLVR
```

Kernel 提供统一的任务、Evaluation、Run 与证据 contracts。Dataset exporter 和
训练系统可以将保留的 Agent trajectories 转换为 SFT、RL 与 distillation 数据，
再将训练完成的 Agent 交回 held-out Evaluation。这样，Environment catalog 既是
Agent Benchmark surface，也是可规模化的可验证长程经验来源。

## 继续阅读

- [核心概念 →](/docs/concepts/)
- [Evaluation 与 Runs →](/docs/evaluation/)
- [环境目录 →](/environments/)
- [Core16 结果 →](/results/)
- [论文 ↗](https://arxiv.org/abs/2607.02440)
