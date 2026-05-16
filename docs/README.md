# HLBench 文档

HLBench 是一个用于评测 Heuristic Learning 能力的 benchmark 设计。

这里的 Heuristic Learning 不是让模型一次性写出一个好策略，而是评测模型在有限实验预算下，能否作为 outer-loop optimizer，通过观察训练反馈、诊断失败、修改代码、运行实验和维护工作区，持续改进一个外部 heuristic system。

HLBench 第一阶段聚焦 Gymnasium 环境，先用轻量控制任务打磨协议，再扩展到 MiniGrid、Box2D、MuJoCo、视觉控制和未来 Atari 套件。环境自己的 episode reward 是主要分数，benchmark 额外记录 success rate、步数、无效 patch、复杂度增长和 regression 等辅助指标。

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
3. [环境支持 Roadmap](benchmark/environment_roadmap.md)
4. [可见性边界](benchmark/visibility_boundary.md)
5. [指标定义](benchmark/metrics.md)
6. [复杂度评估](benchmark/complexity.md)
7. [Workspace 布局](agent_protocol/workspace_layout.md)
8. [无回退迭代](agent_protocol/no_rollback_loop.md)
9. [Gymnasium 适配](environment_protocol/gymnasium_adapter.md)
10. [代码重构设计](code/README.md)

历史规划文档已归档到 [deprecated/](deprecated/)，仅作为上下文保留。

## 一次 run 长什么样

一个典型 HLBench run：

```text
create H_0 workspace
for epoch in 0..K-1:
  expose previous train feedback to agent workspace, or empty feedback at epoch 0
  run agent to inspect files, run allowed train commands, and edit policy.py
  independently evaluate submission on train / validation / heldout
  write transition, event trace, feedback history, checkpoint, and learning curve
  continue from submitted workspace as H_{t+1}
report heldout learning curve and final heldout score
```

关键点：

- agent 只能看到训练集反馈；
- validation 和 heldout 由 benchmark 私有评估，只保留聚合分数；
- validation / heldout 不生成 replay、trace 或 failure details；
- 每轮都记录完整 transition；
- 默认使用 no-rollback 语义，`accepted=false` 只是评估标签，不阻止后续从该 workspace 继续优化。

## 第一版范围

第一版目标是分阶段把 16 环境核心 Gymnasium benchmark 做扎实：

- 固定 observation/action/reward 接口；
- 固定 train / validation / heldout seed split；
- 固定 workspace 和日志协议；
- 固定 event trace 与 transition 数据格式；
- 固定 policy 复杂度评估；
- 固定 HTML 报告与 learning curve；
- 用覆盖低维控制、复杂动力学、程序式导航和连续控制的环境测试持续优化能力和跨任务稳定性。

不一次性把所有环境标为 official。只有当轻量任务上的协议、日志、可见性和指标稳定后，再逐步 officialize Box2D、MiniGrid、MuJoCo、视觉任务，并把 Atari 作为后续扩展套件。
