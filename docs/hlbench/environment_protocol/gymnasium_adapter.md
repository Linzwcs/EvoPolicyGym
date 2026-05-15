# Gymnasium 适配

第一版 HLBench 聚焦 Gymnasium 环境。目标不是覆盖所有 RL 任务，而是用一组轻量、可复现、可快速 rollout 的游戏任务，把 heuristic learning 协议做稳定。

## 环境要求

一个 Gymnasium scenario 至少需要：

- `env_id`
- train / validation / heldout seed split；
- episode 数配置；
- max steps；
- observation 和 action 说明；
- reward 解释；
- success 条件；
- 初始 weak policy；
- rollout summary 生成方式；
- failure mode 分类。

示例：

```json
{
  "scenario_id": "minigrid_keycorridor_s3r2_v0",
  "env_id": "MiniGrid-KeyCorridorS3R2-v0",
  "max_steps": 270,
  "train_episodes": 10,
  "validation_episodes": 30,
  "heldout_episodes": 50
}
```

## Policy 接口

policy 应该是普通 Python 代码，而不是训练好的神经网络权重。

推荐最小接口：

```python
class Policy:
    def reset(self, seed=None):
        pass

    def act(self, observation, reward, terminated, truncated, info):
        return action
```

runner 负责：

- 创建环境；
- reset policy；
- 每步调用 `act`；
- 检查 action 合法性；
- 记录 reward、done、info 和失败摘要。

## Observation 暴露

agent 可以使用环境标准返回的 observation。不要额外暴露 simulator hidden state。

例如 MiniGrid：

- 如果 wrapper 返回局部视野 image，那么 policy 只能基于这个 image 和历史记忆行动；
- 如果 scenario 明确使用 fully observable wrapper，那么 full grid 就是公开 observation；
- wrapper 的选择必须写入 task spec 和 scenario config。

benchmark 不应该在 train 日志里隐藏 policy 实际能看到的 observation，也不应该把 policy 看不到的内部对象作为额外提示。

## Reward

主分数使用游戏自己的 episode return：

```text
episode_return = sum(environment reward over episode)
mean_return = average episode_return over split
```

HLBench 不在单环境内把 reward 强行归一化到 0-1。很多 Gymnasium 环境本身 reward 范围接近 0-1，但这只是环境设计结果，不是 benchmark normalization。

跨环境总榜如果需要聚合，可以后续定义：

```text
normalized_score = (raw_score - random_baseline) / (reference_score - random_baseline)
```

但 raw reward 必须始终保存和报告。

## Seed Split

每个 scenario 固定三组 seed：

```text
train:
  agent 可通过公开工具运行，可看到失败细节和 replay。

validation:
  benchmark 用于过程评估和 accepted 标签。只持久化聚合分数，不生成 replay、trace 或失败细节。

heldout:
  benchmark 用于最终报告。只持久化聚合分数，不生成 replay、trace 或失败细节。
```

train set 应该足够大，让 agent 能观察到多样失败；validation 和 heldout 应该更大，减少偶然 seed hack。

## Rollout Summary

每个 split 的 summary 至少包含：

```json
{
  "env_id": "MiniGrid-KeyCorridorS3R2-v0",
  "split": "train",
  "episodes": 10,
  "mean_return": 0.73,
  "success_rate": 0.8,
  "mean_steps": 72.5,
  "invalid_action_episodes": 0,
  "failure_modes": [
    {"mode": "timeout", "count": 2}
  ]
}
```

train summary 可以复制到 workspace。validation / heldout 只生成 score summary，用于报告和曲线，不复制到 workspace。

validation / heldout 的 score summary 不应包含逐 episode 记录、失败模式、replay 路径、动作序列、observation 序列或 seed 列表。

## Failure Modes

failure mode 应该是环境无关字段和 scenario 特定字段的组合。

通用字段：

- timeout；
- invalid_action；
- no_reward_progress；
- exception；
- truncated；
- low_return。

scenario 特定字段示例：

- cannot_find_key；
- door_not_opened；
- carrying_wrong_object；
- repeated_wall_collision；
- oscillation；
- reached_goal_without_required_object。

failure mode 是 train 诊断辅助。validation / heldout 不应生成 failure mode 明细；它们只报告聚合分数。

## 第一批任务选择

Gymnasium 内优先选择：

- episode 成本低；
- observation/action 容易文档化；
- reward 有清晰解释；
- heuristic policy 可以从零写出来；
- seed 变化能产生足够失败多样性；
- 不需要长时间神经网络训练。

MiniGrid 是合适起点，因为它有明确目标、离散动作、可解释失败和低 rollout 成本。
