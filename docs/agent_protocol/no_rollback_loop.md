# 无回退迭代

HLBench 第一版采用 no-rollback outer loop。

也就是说，每个 epoch 结束后，agent 修改后的 submitted workspace 都会成为下一轮起点：

```text
H_t --agent update--> submitted H_{t+1}
checkpoint submission as H_{t+1}
continue from H_{t+1}
```

即使本轮 `accepted=false`，下一轮也继续从这个 submitted workspace 开始。

## 为什么不 rollback

Heuristic Learning 不是普通超参数搜索，也不是只挑选最优 patch 的离散优化。我们想评测 agent 是否能在一个持续演化的系统里：

- 发现自己上轮方向无效；
- 修复之前引入的问题；
- 从 partial attempt 中继续；
- 删除或简化无效逻辑；
- 把失败经验写入 notes 或 memory；
- 在真实工作区中承担历史修改的后果。

如果每个非提升 patch 都被 runner 回滚，agent 就失去了学习如何维护长期系统状态的压力。

## `accepted` 的含义

`accepted` 不是状态推进条件，而是评估标签。

```text
accepted=true
  本轮在 validation 或配置的选择指标上有足够提升。

accepted=false
  本轮没有达到提升阈值，或复杂度/回归/无效 patch 使 reward 不够好。
```

无论 true 还是 false，都会产生：

```text
transition.json
epoch.json
input/policy.py
submission/policy.py
checkpoint H_{t+1}
```

## Final 与 Best

no-rollback 下必须同时报告：

```text
final_heldout_mean_return
best_heldout_mean_return
```

因为最后一个 checkpoint 可能比历史最好 checkpoint 差。这不是实现 bug，而是 no-rollback 语义下必须显式衡量的风险。

如果 paper 或 leaderboard 只报告 best score，会掩盖 agent 持续维护系统的失败。final score 和 AUC 更能体现真实能力。

## Transition 记录

每轮 transition 应包含：

```json
{
  "epoch": 3,
  "checkpoint_version_before": "H_003",
  "checkpoint_version_after": "H_004",
  "continued": true,
  "accepted": true,
  "state": {},
  "action": {},
  "result": {},
  "reward": {}
}
```

字段要求：

- `continued` 在 no-rollback loop 中恒为 `true`；
- `checkpoint_version_after` 总是递增；
- `accepted=false` 时也必须有 checkpoint；
- `reject_reason` 只解释评估结果，不表示状态没有推进。

## 风险

no-rollback 会带来三个风险：

- bad patch 可能污染后续 workspace；
- policy complexity 可能持续膨胀；
- final checkpoint 可能回退。

因此 benchmark 必须记录：

- complexity growth；
- regression rate；
- invalid patch rate；
- final vs best gap；
- 每轮 policy diff。

这些风险不是要被隐藏，而是 HLBench 需要评测的能力本身。
