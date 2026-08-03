---
id: index
locale: zh
page: documentation
title: 文档
description: 安装 EvoPolicyGym、评估 Program、运行 Coding Agent，或编写 Benchmark。
lead: "评估可执行 Policy，并记录 Coding Agent 如何改进它们。"
index: D0
docsVersion: v0.3
status: current
slug: /
sidebar_position: 1
---

这些页面对应当前 `v0.3` 实现。

## 按任务查找

| 目标 | 文档 |
| --- | --- |
| 评估 CartPole 基线 | [快速开始](./getting-started.md) |
| 打包 Policy 源码 | [Programs](./programs.md) |
| 编写 Policy | [Policy API](./policy.md) |
| 评估一个 Program | [Evaluation](./evaluation.md) |
| 让 Coding Agent 修改 Program | [Runs](./runs.md) |
| 添加 Environment 分发包 | [Benchmark 编写](./authoring.md) |

[核心概念](./concepts.md)说明生命周期和信任边界。运行 Policy 或 Agent 代码前，
请阅读[执行与安全](./runtime.md)。

[环境目录](/environments/)列出可用 Benchmark。[结果档案](/results/)保存历史实验
和重跑；这些结果不代表当前运行时。
