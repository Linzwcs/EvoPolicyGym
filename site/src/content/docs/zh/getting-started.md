---
locale: zh
page: getting-started
section: start
title: "快速开始"
navTitle: "快速开始"
description: "安装 EvoPolicyGym 0.3 并评估 CartPole 参考 Benchmark。"
lead: "安装可移植 Kernel，运行独立分发的 Benchmark，并检查已经提交的 Feedback。"
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## 环境要求

- Python `>=3.12,<3.13`
- [`uv`](https://docs.astral.sh/uv/) `0.11.16`
- 本地仓库 checkout
- 只使用可信的 Policy 与 Agent 代码

> **安全边界。** 当前 `ProcessExecution` setting 会以操作系统用户的权限启动
> 本地子进程。它不是沙箱。

## 安装 Kernel

```console
git clone https://github.com/Linzwcs/EvoPolicyGym
cd EvoPolicyGym
uv sync
uv run evopolicygym --version
```

预期版本输出：

```text
evopolicygym 0.3.0
```

基础 package 包含可移植的 Evaluation 与 Program-evolution Kernel。
具体 Environment 由可独立安装的 Benchmark distribution 提供。

## 安装 CartPole

当前参考 distribution 位于 `environments/cartpole`：

```console
uv sync --project environments/cartpole --extra dev
```

它安装为 `evopolicygym-benchmark-cartpole`，import 名称为
`evopolicygym_cartpole`。

## 评估 baseline

在 5 个确定性的 validation Episodes 上评估 package 中的 baseline：

```console
uv run --project environments/cartpole \
  evopolicygym-cartpole evaluate \
  --episodes 5 \
  --allow-unsafe-process
```

命令输出一个 JSON object，其中包含 Benchmark ID、不可变 Program digest、
标量分数与 Benchmark 定义的 Feedback content。

本地执行没有隔离，因此必须提供确认参数。这个参数不会增加 containment，
也不会改变 execution profile。

## 运行 Coding Agent

完成 Codex CLI 认证后，可以启动一个小规模开发 Run：

```console
uv run --project environments/cartpole \
  evopolicygym-cartpole run \
  --model gpt-5.5 \
  --record-to runs/cartpole-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --max-episodes-per-submission 10 \
  --allow-unsafe-process
```

Agent 只能编辑 `runs/cartpole-001/workspace/program/`。已经提交的公开
Feedback 会写入相邻的 `workspace/feedback/`。Host 侧 Programs、artifacts、
events 与 Agent logs 分开保留。

## 刚才发生了什么

1. 初始 Policy 目录成为不可变、内容寻址的 `Program`。
2. Coding Agent 获得固定 workspace、Benchmark specification 与有限提交权限。
3. 每次 Evaluation 都会规划确定性的 Episodes。
4. 每个 Episode 都创建全新的 Environment 与 Policy 进程。
5. 完成的 Submission 原子发布 Program、Feedback、Episode summaries 与可选 artifacts。
6. Agent 从完全发布的 Submissions 中选择最终 Program。

## 下一步

- [阅读核心概念 →](../concepts/)
- [阅读 Policy ABI →](../policy/)
- [理解 Evaluation 与 Runs →](../evaluation/)
- [查看环境目录 →](../../environments/)
