---
id: index
locale: zh
page: documentation
title: 文档
description: 开始使用 EvoPolicyGym，理解评估模型，并编写独立的 Benchmark distribution。
lead: "用于评估 Coding Agent 如何将有界 Environment Feedback 转化为可执行 Policy 系统的研究软件。"
index: D0
docsVersion: v0.3
status: current
slug: /
sidebar_position: 1
---

文档跟随当前 `v0.3` 实现，说明公开 Python SDK、有界 Policy ABI、Evaluation 与
Run 语义、进程执行限制，以及公开的 Benchmark authoring surface。

## 从这里开始

- [快速开始](./getting-started.md)介绍如何安装 Kernel 并评估内置的 CartPole baseline。
- [核心概念](./concepts.md)解释 Program、Policy、Submission、Feedback、
  Validation 与 Assessment。

## 核心参考

- [Policy ABI](./policy.md)定义 Observation、Action、同 Episode 状态与失败行为。
- [Evaluation 与 Runs](./evaluation.md)定义公开搜索、选择和 held-out 测量生命周期。
- [Runtime 与安全](./runtime.md)记录进程执行方式及其限制。

## 扩展 EvoPolicyGym

- [Benchmark 编写](./authoring.md)说明独立安装 Environment distribution
  所使用的公开 conformance surface。

[Environment 目录](/environments/)记录当前 Benchmark surface。历史实验分数与
重跑保存在[结果档案](/results/)中，并与当前 runtime 明确区分。
