---
locale: zh
page: getting-started
section: start
title: "快速开始"
navTitle: "快速开始"
description: "安装 EvoPolicyGym 0.3，并完成第一次 Evaluation 或 Program-evolution Run。"
lead: "安装 Kernel，评估一个示例 Program，并启动一次有界的 Coding Agent Run。"
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## 环境要求

- Python `>=3.12,<3.13`
- [`uv`](https://docs.astral.sh/uv/) `0.11.16`
- 本地仓库 checkout
- 可信的 Policy 与 Agent 代码

> `ProcessExecution` 以当前操作系统用户的权限启动 Policy 与 Agent 子进程。

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

Kernel 提供 Evaluation 与 Program-evolution 生命周期。Benchmark distribution
提供 Environment、Policy 契约与评分。

## 选择 Benchmark

从[环境目录](../../environments/)选择 distribution。下面的命令使用体积较小的
CartPole distribution：

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
```

## 完成一次 Evaluation

在 5 个确定性的 validation Episodes 上评估示例 distribution 提供的 Program：

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole evaluate \
  --episodes 5 \
  --allow-unsafe-process
```

命令输出一个 JSON object，其中包含 Benchmark identity、不可变 Program digest、
标量分数与公开 Feedback。Feedback 内容遵循所选 Benchmark 契约。

`--allow-unsafe-process` 用于确认以当前本地用户权限执行。

## 运行 Coding Agent（可选）

完成 Codex CLI 认证后，让 Agent 在较小的开发预算内修改示例 Program：

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole run \
  --model gpt-5.5 \
  --record-to runs/quickstart-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --allow-unsafe-process
```

Agent 自行决定每次 Submission 使用的 Episode 数量。Program workspace 位于
`runs/quickstart-001/workspace/program/`，已提交 Feedback 位于
`workspace/feedback/`，Host 记录保存在 Run 目录中。

## Run 执行的流程

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
- [选择并配置 Environment →](../../environments/)
