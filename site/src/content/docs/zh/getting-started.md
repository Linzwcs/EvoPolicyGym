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
uv run --project environments/gymnasium/classic_control/cartpole python - <<'PY'
from cartpole import CartPoleBenchmark, baseline_program
from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.execution import ProcessExecution

result = evaluate(
    baseline_program(),
    CartPoleBenchmark(),
    execution=ProcessExecution.unsafe(),
    config=EvaluationConfig(split="validation", episodes=5, seed=42),
)
print(result.feedback.score)
print(result.feedback.content)
PY
```

示例输出标量分数与 Benchmark 定义的公开 Feedback。`EvaluationResult` 还保留
Benchmark identity、不可变 Program digest 与经过净化的 Episode summaries。

`ProcessExecution.unsafe()` 明确确认以当前本地用户权限执行。

## 运行 Coding Agent（可选）

完成 Codex CLI 认证后，让 Agent 在较小的开发预算内修改示例 Program：

```console
mkdir -p runs
uv run --project environments/gymnasium/classic_control/cartpole \
  python scripts/run_cartpole_codex.py \
  --model gpt-5.6-luna \
  --reasoning-effort high \
  --record-to runs/quickstart-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --episode-pool-size 60 \
  --max-episodes-per-submission 10 \
  --allow-unsafe-process
```

Host 会在 Agent 启动前构建 60 个固定的 Run-local 训练 Episode identity。Agent
为每次 Submission 选择非空编号集合，同时遵守总计 30 个 Episode 单位、单次至多
10 个的预算。Program workspace 位于
`runs/quickstart-001/workspace/program/`，已提交 Feedback 位于
`workspace/feedback/`，Host 记录保存在 Run 目录中。

在有效的 Agent Session 内，Submission 使用单点编号与半开区间：

```console
evopolicygym submit program --episodes "0:2,4:8"
```

上述 selector 会评估编号 `0, 1, 4, 5, 6, 7`。在后续 Submission 中复用编号可以
得到相同的 Episode specification 与 Policy seed，用于配对比较；但每次使用仍会
创建全新的 Environment 与 Policy runtime，并再次扣除预算。真实 seed 始终隐藏。

## Run 执行的流程

1. 初始 Policy 目录成为不可变、内容寻址的 `Program`。
2. Agent 执行前，Host 根据 Run seed 构建一个确定性的编号训练池。
3. Coding Agent 获得固定 workspace、Benchmark specification、可选池边界，以及
   有限的 Submission 与 Episode 权限。
4. 每次 Submission 显式选择池编号；每个所选编号都创建全新的 Environment 与
   Policy 进程。
5. 完成的 Submission 原子发布 Program、所选编号、Feedback、Episode summaries
   与可选 artifacts。
6. Agent 从完全发布的 Submissions 中选择最终 Program。

## 下一步

- [阅读核心概念 →](../concepts/)
- [阅读 Policy ABI →](../policy/)
- [理解 Evaluation 与 Runs →](../evaluation/)
- [选择并配置 Environment →](../../environments/)
