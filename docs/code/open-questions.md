# Open Questions

本文件记录尚未完全定死的需求和设计问题。决策完成后，应同步回 `requirements.md`、`schema-design.md` 或对应协议文档。

## P0: 实现前必须定

### Official Scenario 准入

当前规则已移入 [scenario-validation.md](scenario-validation.md)。

未决：

- 是否允许 reward scale 不归一化。

### Public / Private 数据矩阵

当前规则已移入 [visibility-and-data-boundary.md](visibility-and-data-boundary.md)。

未决：

- private config 是否可落盘，落盘时如何隔离。
- report 是否默认显示 validation 指标，还是只显示 heldout aggregate。

### Policy Context

当前倾向：

```python
context = {
  "step": 0,
  "max_steps": 500,
  "action_schema": {},
  "last_reward": 0.0,
  "terminated": false,
  "truncated": false
}
```

未决：

- 是否给 policy 每步传 `last_reward`。
- 是否把完整 `task_contract` 传给 policy，还是只传 action schema。
- 是否允许 policy 在 `reset` 中读取 task contract 文件。

### Action Validation

当前倾向：

- `Discrete`: int in `[0, n)`.
- `Box`: numeric array with shape and range.
- `MultiDiscrete`: integer array with `nvec` bounds.
- `MultiBinary`: binary array.
- `Dict` / `Tuple`: recursive validation.
- invalid action 直接触发本轮 minimum score。

未决：

- continuous action 是否允许自动 cast dtype。
- floating point range 是否允许 epsilon tolerance。

## P1: 重构中需要定

### Agent Budget

当前倾向：

- 每个 epoch 有 wall clock timeout。
- 可选限制 train rollout 次数。
- stdout/stderr 和 artifact 大小应有限制。

未决：

- 命令次数是否硬限制。
- train rollout budget 是通过 wrapper 计数，还是通过 artifact 检查推断。

### Contract Violation Severity

当前倾向：

- 修改只读文件: minimum score。
- 修改 task/feedback/evaluator: minimum score，标记 severe violation。
- 访问 private data: minimum score，并可能 run disqualification。

未决：

- severe violation 是否总是 run disqualification，还是只有 private data access 才 disqualify。
- 是否继续评估 minimum score 后的 policy 以便诊断。

### Reward Normalization

当前倾向：

- summary 保留环境原始 return。
- reward calculator 可按 scenario 配置归一化。

未决：

- benchmark 排名使用 normalized heldout AUC 还是 raw heldout return。
- 不同 reward scale 的环境是否能放进同一 leaderboard。

## P2: 可以稍后定

### Report Format

未决：

- HTML report 的最小字段。
- 是否包含 train replay 可视化。
- 是否为每个 epoch 链接完整 transition。
- validation 曲线是否默认显示，还是放进高级诊断区。
- static HTML 是否必须完全自包含。

### Experiment Tracking

当前倾向：

- agent-created train experiment 必须只使用 train split。
- 每个 agent-created train experiment 写 `experiment.json`。
- benchmark standard evaluation 是正式评分依据。
- human-facing report 可以汇总 agent-created train experiment，但不回流给 agent。

未决：

- agent-created train experiment 是否计入硬 rollout budget。
- agent 手写的 notes 是否进入 final report。
- diagnostic evaluation 是否允许在 minimum score 后运行。

### Prompt Protocol

当前倾向：

- prompt 只指导 agent 阅读 workspace 文件。
- final JSON 是 best effort，不作为成功必要条件。

未决：

- 是否定义统一 final JSON schema。
- 是否要求 agent 报告 commands run 和 edited files。

### Example Scenarios

需要补完整例子：

- `CartPole-v1`: 通用 Gymnasium、array observation、Discrete action。
- `MiniGrid-DoorKey`: Dict observation、mission 字段、MiniGrid action meanings。
