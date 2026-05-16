# 问题形式化

HLBench 把 Heuristic Learning 建模为预算受限的外层优化问题。

模型的任务不是直接输出一个最终答案，而是在一个可执行反馈环境中反复维护 heuristic system。每一轮它读取当前工作区和训练反馈，决定下一步实验和代码修改，并把系统从 `H_t` 推进到 `H_{t+1}`。

## 三层系统

```text
inner environment E
  游戏或任务本身。它定义 observation、action、transition、reward、done。

heuristic system H_t
  外部可执行策略系统。最小版本包括 policy.py；完整版本还包括 notes、memory、回归用例和训练 rollout 日志。

learner pi_theta
  执行 heuristic learning 的模型或 coding agent。它观察 workspace 和 train feedback，然后修改 H_t。
```

HLBench 评测的是 learner 的 outer-loop 能力：

```text
pi_theta: (H_t, train feedback, budget) -> update action
```

而不是某个 policy 的单次生成能力。

## Outer-Loop POMDP

一次 epoch 可以写成：

```text
s_t = {
  task_spec,
  workspace_t,
  policy_code_t,
  public_train_rollout_summary_t,
  public_train_failures_t,
  previous_epoch_summaries_t,
  budget_remaining
}

a_t = {
  command_trajectory,
  diagnosis,
  code_diff,
  notes_or_memory_update,
  final_report
}

o_{t+1} = rollout(H_t + a_t, E)

r_t = reward(o_{t+1}, o_t, cost_t, violations_t)

H_{t+1} = apply(H_t, a_t)
```

这是 POMDP，因为 learner 看不到完整任务分布、私有 seed、validation / heldout failure details、evaluator source、最优策略或真实 simulator hidden state。它只能通过训练反馈和自己选择的实验间接推断如何改进。

## Heuristic System

最小可运行的 heuristic system：

```text
H_t = {
  system/policy.py,
  task.md,
  train rollout summaries,
  agent-authored tools
}
```

更完整的 heuristic system：

```text
H_t = {
  system/policy.py,
  system/helper code,
  task.md,
  feedback/current/,
  feedback/history/,
  tools/,
  experiments/
}
```

HLBench 不强制第一版实现所有组件，但文档和数据格式应该允许未来扩展。核心原则是：agent 维护的是一个可继续演化的系统，而不是只写一个孤立函数。

## Action Space

不要把 action 简化成一个 patch。真实 agent 的 action 包括：

- 读哪些文件；
- 运行哪些 train rollout；
- 如何总结失败；
- 修改哪段 policy；
- 是否添加 notes 或 memory；
- 是否简化已有规则；
- 如何解释预期收益和风险。

所以完整 action 更接近：

```text
a_t = command trajectory + file edits + final structured report
```

训练数据可以压缩成 `state -> final patch`，但 benchmark 评测和审计应该保留 event trace。

## 目标函数

HLBench 关注 learning curve，而不是只看最后一次分数：

```text
J(pi_theta) = E [
  AUC(heldout_return over epoch budget)
  + final_heldout_return
  - rollout_cost
  - invalid_patch_penalty
  - regression_penalty
  - complexity_penalty
]
```

最重要的问题是：

```text
在同样实验预算下，agent 能不能更快、更稳地让 H_t 变好？
```

这也是 HLBench 与普通 coding benchmark、单步 program synthesis、AutoML search 的核心区别之一。

## 能力维度

HLBench 至少评测六类能力：

- state understanding：读懂 observation/action、policy 结构和训练反馈；
- failure diagnosis：把失败归因到具体机制；
- experiment selection：知道该跑哪些 train 实验，不浪费预算；
- patch generation：写出局部、可执行、有效的 heuristic 修改；
- regression control：提升新行为时不破坏已有能力；
- transfer：从某些任务学到的改进行为能迁移到新任务。

这些能力共同构成 “learning beyond gradients” 风格的外层学习能力。
