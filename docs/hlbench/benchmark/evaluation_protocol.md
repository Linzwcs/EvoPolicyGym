# 评估流程

HLBench 的评估单位是一个多 epoch run。每个 epoch 是一次 heuristic system transition：

```text
H_t -> H_{t+1}
```

benchmark 负责隔离环境、运行 rollout、计算指标、维护 checkpoint 和记录数据；agent 负责在可见 workspace 中做实验、诊断和修改。

## Run 初始化

每个 run 开始时，benchmark 创建一个 learner workspace：

```text
runs/<run_id>/workspace/
  AGENTS.md
  task.md
  system/
  feedback/
  tools/
  experiments/
```

其中：

- `task.md` 描述任务、动作空间、观测接口、目标、policy 接口和允许的 train rollout 命令；
- `system/` 是当前 heuristic system，第一版核心是 `system/policy.py`；
- `feedback/` 保存 benchmark 生成的公开 train feedback，agent 只读；
- `tools/` 初始可以为空，agent 可以在这里编写分析工具；
- `experiments/` 是 agent 自己保存临时实验输出和 notes 的区域。

runner metadata、event trace、checkpoint index、validation/heldout 聚合分数不放进 workspace。它们由 run directory 管理。

初始化时还会创建：

```text
runs/<run_id>/checkpoints/H_000/workspace/
runs/<run_id>/events.jsonl
runs/<run_id>/transitions.jsonl
runs/<run_id>/learning_curve.json
```

## Epoch 流程

每个 epoch 执行：

```text
1. evaluate H_t on train split
2. copy public train feedback into workspace
3. build prompt for agent
4. run agent inside workspace
5. verify protected files
6. compile/import candidate policy
7. privately evaluate before and after policies on validation / heldout
8. evaluate candidate on train
9. compute reward components
10. write transition and epoch summary
11. checkpoint candidate workspace as H_{t+1}
```

注意：validation 和 heldout 的 before/after 评估由 benchmark 在 workspace 外部完成。它们只持久化聚合分数，用于报告、learning curve 和 reward 计算；不生成 replay、trace、per-episode logs 或 failure details。

## Public Train Feedback

agent 可以看到训练 split 的反馈，例如：

```text
workspace/feedback/current/summary.json
workspace/feedback/current/episodes.jsonl
workspace/feedback/current/failures.jsonl
workspace/feedback/current/replays/
workspace/feedback/history/epoch_003/summary.json
```

第一版可以给 agent 公开 train summary、失败摘要和 replay 文件。benchmark 标准反馈放在 `feedback/` 下，agent 自己额外运行的临时实验放在 `experiments/` 下。是否分析 replay、是否写自己的调试脚本、是否维护 notes，应该由 agent 自己决定。

## Private Evaluation

benchmark 私有执行：

```text
validation before / after
heldout before / after
```

私有评估的聚合分数只进入：

- `events.jsonl`
- `transition.json`
- `epoch_summary.json`
- `learning_curve.json`
- final report

它们不复制到 learner workspace，也不作为下一轮 prompt 的可见反馈。validation / heldout 不能写出 replay、trace、`episodes.jsonl` 或 `failures.jsonl`。

推荐私有分数字段：

```json
{
  "split": "heldout",
  "episodes": 50,
  "mean_return": 0.93,
  "success_rate": 1.0,
  "mean_steps": 20.6
}
```

不要包含 seed、逐 episode 记录、失败模式、动作序列、observation 序列或可还原环境状态的信息。

## Checkpoint 语义

每个 epoch 都产生 checkpoint：

```text
runs/<run_id>/checkpoints/H_001/workspace/
runs/<run_id>/checkpoints/H_002/workspace/
...
```

`H_000` 是初始 workspace。第 `t` 轮 agent 修改后的 workspace 保存为 `H_{t+1}`。

第一版 HLBench 使用 no-rollback 语义：无论本轮是否提升 validation，candidate workspace 都成为下一轮起点。`accepted` 是评估标签，不是状态推进条件。

## 输出文件

run-level artifacts：

```text
runs/<run_id>/events.jsonl
runs/<run_id>/transitions.jsonl
runs/<run_id>/learning_curve.json
runs/<run_id>/report/report.html
runs/<run_id>/report/learning_curve.svg
runs/<run_id>/report/metrics.json
```

per-epoch artifacts：

```text
runs/<run_id>/steps/epoch_000/prompt.md
runs/<run_id>/steps/epoch_000/epoch_summary.json
runs/<run_id>/steps/epoch_000/transition.json
runs/<run_id>/steps/epoch_000/policy.patch
runs/<run_id>/steps/epoch_000/codex/stdout.txt
runs/<run_id>/steps/epoch_000/codex/stderr.txt
runs/<run_id>/steps/epoch_000/codex/final.json
runs/<run_id>/steps/epoch_000/evaluator/
```

## 最终报告

最终报告至少包含：

- env id 和 scenario id；
- epoch 数；
- train / validation / heldout learning curve；
- final heldout mean return；
- best heldout mean return；
- heldout AUC；
- accepted / rejected epoch 数；
- invalid patch rate；
- policy complexity growth；
- 每轮 transition 链接。

主指标使用 heldout learning curve 和 final heldout return。validation 用于过程诊断，train 用于 agent 可见反馈，不应该作为最终能力声明的主证据。
