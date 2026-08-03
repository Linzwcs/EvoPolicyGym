---
locale: zh
page: getting-started
section: start
title: "快速开始"
navTitle: "快速开始"
description: "安装 EvoPolicyGym 0.3，并评估 CartPole 基线。"
lead: "安装一个 Benchmark，完成一次确定性 Evaluation。"
index: D1
order: 1
docsVersion: v0.3
status: draft
---

## 安装

需要 Python `3.12` 和 [`uv`](https://docs.astral.sh/uv/) `0.11.16`。

```console
git clone https://github.com/Linzwcs/EvoPolicyGym
cd EvoPolicyGym
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym --version
```

预期版本输出：

```text
evopolicygym 0.3.0
```

该命令会安装 EvoPolicyGym 内核和独立的 CartPole Benchmark。

## 评估基线

在 5 个 validation Episodes 上运行内置基线：

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

结果包含标量分数、公开 Feedback、Program 摘要和经过净化的 Episode 结果。

:::warning 本地进程执行

`ProcessExecution.unsafe()` 以当前操作系统用户权限运行 Policy 代码。它不是沙箱，
只应用于可信代码。

:::

## 下一步

- [创建 Program](./programs.md)
- [编写 Policy](./policy.md)
- [配置 Evaluation](./evaluation.md)
- [运行 Coding Agent](./runs.md)
- [选择其他 Environment](/environments/)
