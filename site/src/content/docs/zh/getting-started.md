---
locale: zh
page: getting-started
section: start
title: "快速开始"
navTitle: "快速开始"
description: "安装 EvoPolicyGym 0.3，并运行一个当前控制或游戏 Benchmark。"
lead: "安装可移植 Kernel，选择一个独立分发的 Benchmark，并检查已经提交的 Feedback 或语义回放。"
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

## 安装 Benchmark

CartPole 是轻量参考 distribution，Acrobot 增加稀疏奖励摆起控制，两个 Mountain
Car distribution 对比离散与连续控制，Pendulum 补全 Classic Control 集合，
Balatro 则是长时程游戏 distribution：

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv sync --project environments/gymnasium/classic_control/acrobot --extra dev
uv sync --project environments/gymnasium/classic_control/mountain_car --extra dev
uv sync --project environments/gymnasium/classic_control/mountain_car_continuous --extra dev
uv sync --project environments/gymnasium/classic_control/pendulum --extra dev
uv sync --project environments/jackdaw/balatro --extra dev
```

它们分别安装为 `evopolicygym-benchmark-cartpole`、
`evopolicygym-benchmark-acrobot`、`evopolicygym-benchmark-mountain-car`、
`evopolicygym-benchmark-mountain-car-continuous`、
`evopolicygym-benchmark-pendulum` 与
`evopolicygym-benchmark-balatro`，公开 import package 分别为 `cartpole`、
`acrobot`、`mountain_car`、`mountain_car_continuous`、`pendulum` 与
`balatro`。

## 评估 baseline

在 5 个确定性的 validation Episodes 上评估 package 中的 baseline：

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole evaluate \
  --episodes 5 \
  --allow-unsafe-process
```

命令输出一个 JSON object，其中包含 Benchmark ID、不可变 Program digest、
标量分数与 Benchmark 定义的 Feedback content。

本地执行没有隔离，因此必须提供确认参数。这个参数不会增加 containment，
也不会改变 execution profile。

Acrobot 与 Mountain Car 同样使用 Episode 平均回报，但正常 reward 为非正数。
因此 Policy failure 计入任务的完整 Episode 下限，而不是零分：Acrobot 为 `-500`，
Mountain Car 为 `-200`。两个 distribution 都在 Observation 跨越 Policy 边界前，
把 Gymnasium 数组转换为具名语义字典。

Continuous Mountain Car 使用 `-1.0` 至 `1.0` 的有限浮点 Action。它的零力
baseline 没有成功且回报为 `0`，速度方向策略则约为 `89`。Policy failure 计为
`-100`，低于完整 Episode 的理论最小回报。

Pendulum 固定运行 200 步，没有成功 termination。它的 reward 是角度、角速度与
扭矩代价的负值，最高为零。Policy failure 计为 `-3300`，低于完整 Episode 的
理论最小回报。

## 查看 Balatro

Balatro Benchmark 的每个 Episode 都是一局完整的红色牌组、白注 run。评分为
胜利奖励 1000 分，再加每个已通过 Blind 1 分。Policy 需要处理出牌、弃牌、
Blind 选择、商店、Joker、消耗品、补充包与 Ante。

公开的 `replay.jsonl` artifact 保留每个已收录 step 中 Policy 实际收到的完整
语义 Observation。站点播放器只渲染其中适合阅读的部分，不会缩减底层 artifact。

- [阅读 Balatro Benchmark 契约 →](../../environments/#balatro)
- [打开 Baseline 游戏回放 →](../../environments/balatro/replay/)

## 运行 Coding Agent

完成 Codex CLI 认证后，可以启动一个小规模开发 Run：

```console
uv run --project environments/gymnasium/classic_control/cartpole \
  evopolicygym-cartpole run \
  --model gpt-5.5 \
  --record-to runs/cartpole-001 \
  --max-submissions 3 \
  --episode-budget 30 \
  --allow-unsafe-process
```

默认由 Agent 决定每次 Submission 的 Episode 数量。只有需要额外限制时才添加
`--max-episodes-per-submission N`。

Balatro 还发布了可选的 Policy 优化 skill。它默认关闭；如果希望 Run 将其以只读
形式提供到 `workspace/skill/SKILL.md`，可在调用
`scripts/run_balatro_codex.py` 时传入 `--benchmark-skill`。

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
