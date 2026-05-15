# 指标定义

HLBench 的核心指标是 learning curve。我们关心 agent 在固定预算内如何改进 heuristic system，而不是只看某一轮是否写出好 policy。

## 主指标

### Heldout Learning AUC

```text
heldout_return_auc = AUC(heldout_mean_return over epochs)
```

它衡量 agent 在整个预算过程中的学习效率。

两个 agent 最终分数相同，但一个在第 2 轮就达到高分，另一个第 10 轮才达到高分，前者应该更强。AUC 能体现这一点。

### Final Heldout Mean Return

```text
final_heldout_mean_return
```

最终 checkpoint 在 heldout split 上的平均游戏 reward。

第一版中，这个分数直接使用游戏环境自己的 reward，不做额外归一化。不同环境之间做总榜时，可以后续再定义 per-env normalization，但单环境报告必须保留 raw reward。

### Best Heldout Mean Return

```text
best_heldout_mean_return
```

run 内所有 checkpoint 的最高 heldout mean return。

在 no-rollback 语义下，final checkpoint 可能不是 best checkpoint。因此 final 和 best 都要报告。

## 辅助指标

### Success Rate

```text
success_rate = successful_episodes / total_episodes
```

success 的定义由 scenario 指定。例如 MiniGrid 中通常可以用 episode 是否成功终止、是否拿到正 reward、或 `info["success"]` 判断。

success rate 适合解释策略是否已经学会完成任务，但不一定能区分效率。两个 policy 都 100% success 时，mean return 和 mean steps 更有区分度。

### Mean Steps

```text
mean_steps = average episode length
```

在很多 Gymnasium 游戏里，reward 会惩罚步数或受步数影响。mean steps 是解释 reward 变化的重要辅助指标。

### First Improvement Latency

```text
first_improvement_latency = first epoch where heldout_mean_return > initial_heldout_mean_return
```

它衡量 agent 多快从初始 weak policy 找到有效方向。

### Invalid Patch Rate

```text
invalid_patch_rate = invalid epochs / total epochs
```

invalid 包括：

- patch 语法错误；
- policy import 失败；
- action 不合法；
- 修改 protected files；
- rollout 超时；
- 输出 schema 不合法。

### Regression Rate

```text
regression_rate = epochs with significant validation or heldout drop / total epochs
```

regression 的阈值应由 scenario 或全局配置定义。小的随机波动不应该算 regression。

### Rollout Efficiency

```text
rollout_efficiency = heldout improvement / rollout cost
```

rollout cost 可以按 episode 数、环境 step 数、wall time 或 agent token cost 计算。第一版至少记录 episode 数和 wall time。

### Complexity Growth

```text
complexity_growth = complexity(policy_after) - complexity(policy_before)
```

复杂度使用传统软件工程指标，评估范围只包括 `workspace/system/` 中的可执行策略代码，不包括 `feedback/`、`tools/`、`experiments/` 或 benchmark runner。

第一版推荐记录：

- SLOC / logical LOC；
- file / function / class count；
- cyclomatic complexity mean / max；
- max function length；
- max nesting depth；
- import count；
- patch added / deleted lines。

复杂度不是越低越好。它的作用是惩罚无收益的膨胀，并帮助识别不可维护的策略。

更完整的定义见 [复杂度评估](complexity.md)。

## Reward Components

训练或选择 reward 可以组合：

```text
reward = train_delta
       + validation_delta
       + heldout_delta_for_reporting_or_offline_label
       - invalid_patch_penalty
       - regression_penalty
       - complexity_penalty
       - rollout_cost_penalty
```

注意：

- agent 不应该看到 validation / heldout failure details；
- heldout delta 可以用于离线分析和最终报告；
- 如果 reward 用于在线选择，应谨慎避免 heldout leakage；
- 第一版可以用 validation 决策、heldout 报告。

## 报告要求

每个 run 至少报告：

- `heldout_return_auc`
- `final_heldout_mean_return`
- `best_heldout_mean_return`
- `final_validation_mean_return`
- `final_train_mean_return`
- `accepted_epochs`
- `rejected_epochs`
- `invalid_patch_rate`
- `complexity_growth`
- `final_policy_lines`
- `final_cyclomatic_complexity_max`
- `compression_epochs`

对 benchmark paper 或 leaderboard，主表应该优先展示 heldout AUC 和 final heldout raw reward。
