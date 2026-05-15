# HLBench 文档

HLBench 是一个用于评测 Heuristic Learning 能力的 benchmark 设计。

这里的 Heuristic Learning 不是让模型一次性写出一个好策略，而是评测模型在有限实验预算下，能否作为 outer-loop optimizer，通过观察训练反馈、诊断失败、修改代码、运行实验和维护工作区，持续改进一个外部 heuristic system。

第一版 HLBench 聚焦轻量 Gymnasium 游戏环境，尤其是 MiniGrid 一类可快速复现实验的任务。环境自己的 episode reward 是主要分数，benchmark 额外记录 success rate、步数、无效 patch、复杂度增长和 regression 等辅助指标。

## 核心对象

HLBench 中有三层对象：

```text
inner environment E
  真实任务环境，例如 Gymnasium / MiniGrid 游戏。

heuristic system H_t
  当前可执行 heuristic 系统，包括 policy.py、notes、memory、训练日志和可维护状态。

learner / agent pi_theta
  读 workspace、看 train feedback、运行允许命令、修改 H_t 的模型或 coding agent。
```

benchmark 评测的不是某个静态 policy 的水平，而是：

```text
H_0 -> H_1 -> ... -> H_K
```

也就是模型能否在固定 epoch、固定 rollout budget、固定可见性边界下，让 heuristic system 更快、更稳地变强。

## 推荐阅读顺序

1. [问题形式化](concept/problem_formulation.md)
2. [评估流程](benchmark/evaluation_protocol.md)
3. [可见性边界](benchmark/visibility_boundary.md)
4. [指标定义](benchmark/metrics.md)
5. [复杂度评估](benchmark/complexity.md)
6. [Workspace 布局](agent_protocol/workspace_layout.md)
7. [无回退迭代](agent_protocol/no_rollback_loop.md)
8. [Gymnasium 适配](environment_protocol/gymnasium_adapter.md)

## 一次 run 长什么样

一个典型 HLBench run：

```text
create H_0 workspace
for epoch in 0..K-1:
  run public train rollout for current H_t
  expose train feedback to agent workspace
  run agent to inspect files, run allowed train commands, and edit policy.py
  independently evaluate candidate on train / validation / heldout
  write transition, event trace, checkpoint, and learning curve
  continue from candidate workspace as H_{t+1}
report heldout learning curve and final heldout score
```

关键点：

- agent 只能看到训练集反馈；
- validation 和 heldout 由 benchmark 私有评估，只保留聚合分数；
- validation / heldout 不生成 replay、trace 或 failure details；
- 每轮都记录完整 transition；
- 默认使用 no-rollback 语义，`accepted=false` 只是评估标签，不阻止后续从该 workspace 继续优化。

## 第一版范围

第一版只需要把 Gymnasium benchmark 做扎实：

- 固定 observation/action/reward 接口；
- 固定 train / validation / heldout seed split；
- 固定 workspace 和日志协议；
- 固定 event trace 与 transition 数据格式；
- 固定 policy 复杂度评估；
- 固定 HTML 报告与 learning curve；
- 用多个轻量游戏测试持续优化能力和跨任务稳定性。

不急于引入复杂视觉游戏、MuJoCo、大规模训练或 API RFT。只有当轻量任务上的协议、日志、可见性和指标稳定后，再扩展任务集合。
