# 评估流程

HLBench 的评估单位是一个多 epoch run。每个 epoch 是一次 heuristic system transition：

```text
H_t -> H_{t+1}
```

benchmark 负责隔离环境、运行 rollout、计算指标、维护 checkpoint 和记录数据；agent 负责在可见 workspace 中做实验、诊断和修改。

## Run 初始化

HLBench run directory 使用模型和环境分层：

```text
runs/<model_name>/<env_name>/<run_id>/
```

其中：

- `<model_name>` 是 agent/model/backend 标识，例如 `codex`、`claude`、`local-script`；
- `<env_name>` 是 scenario/env 标识，例如 `mountain_car`、`cartpole_balance`；
- `<run_id>` 是本次实验编号，例如 timestamp、短 SHA 或人工命名。

这些路径段必须是 filesystem-safe slug：只使用小写字母、数字、`-` 和 `_`，不得包含空格、`/`、模型版本中的路径分隔符或 shell 特殊字符。真实模型名称可以额外写入 `run.json` metadata。

每个 run 开始时，benchmark 创建一个 learner workspace：

```text
runs/<model_name>/<env_name>/<run_id>/workspace/
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
- `feedback/` 保存 benchmark 生成的 train feedback，以及历史 aggregate validation score，agent 只读；
- `tools/` 初始可以为空，agent 可以在这里编写分析工具；
- `experiments/` 是 agent 自己保存临时实验输出和 notes 的区域。

runner metadata、event trace、checkpoint index 和 heldout 聚合分数不放进 workspace。validation aggregate 可以进入 `feedback/history/`，但 validation replay、per-episode 记录、failure details 和 seeds 仍不进入 workspace。

初始化时还会创建：

```text
runs/<model_name>/<env_name>/<run_id>/checkpoints/H_000/workspace/
runs/<model_name>/<env_name>/<run_id>/events.jsonl
runs/<model_name>/<env_name>/<run_id>/transitions.jsonl
runs/<model_name>/<env_name>/<run_id>/learning_curve.json
```

## Epoch 流程

每个 epoch 执行：

```text
1. publish previous epoch train feedback into workspace, or an empty current feedback marker at epoch 0
2. run agent inside workspace
3. let agent inspect files, run allowed train-only rollouts, and edit `system/policy.py`
4. verify protected files
5. compile/import submitted policy
6. evaluate submission on public train split
7. evaluate submission on validation / heldout splits
8. compute reward against the previous aggregate baseline, or minimum-score baseline at epoch 0
9. write transition, epoch summary, feedback history, and latest train feedback
10. continue from submitted workspace as H_{t+1}
```

注意：初始 policy 不预先运行。每轮先让 agent 产生 submission，再由 benchmark 在 workspace 外部评估 submission。validation 和 heldout 只生成并持久化聚合分数，用于报告、learning curve 和 reward 计算；不得生成 replay、trace、per-episode logs 或 failure details。

## Public Train Feedback

agent 可以看到训练 split 的反馈，例如：

```text
workspace/feedback/current/summary.json
workspace/feedback/current/episodes.jsonl
workspace/feedback/current/failures.jsonl
workspace/feedback/current/replays/
workspace/feedback/history/epoch_003/train/summary.json
workspace/feedback/history/epoch_003/train/episodes.jsonl
workspace/feedback/history/epoch_003/train/replays/
workspace/feedback/history/epoch_003/validation_summary.json
```

第一版可以给 agent 公开 train summary、失败摘要和 replay 文件。历史目录也可以暴露 aggregate validation score，帮助 agent 看迭代趋势。benchmark 标准反馈放在 `feedback/` 下，agent 自己额外运行的临时实验放在 `experiments/` 下。是否分析 replay、是否写自己的调试脚本、是否维护 notes，应该由 agent 自己决定。

## Private Evaluation

benchmark 执行标准评估：

```text
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/
  input/policy.py
  submission/policy.py
  evaluation/train/
  evaluation/validation/
  evaluation/heldout/
```

私有评估的聚合分数只进入：

- `events.jsonl`
- `transition.json`
- `epoch.json`
- `learning_curve.json`
- final report

Validation aggregate summaries may be copied into `workspace/feedback/history/`.
Heldout summaries are not copied to learner workspace. validation / heldout
evaluation 不能持久化 replay、trace、`episodes.jsonl`、`failures.jsonl`
或 frame replay。Image-observation scenarios may use ephemeral current-frame
files as the policy-visible observation transport during evaluation, but these
files must be deleted before the private split result is returned and must not
be referenced from any private summary or report artifact.

Train evaluation is stored separately under `evaluation/train/`. It may
contain replay and per-episode artifacts because it is public train feedback,
not private evaluation.

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
runs/<model_name>/<env_name>/<run_id>/checkpoints/H_001/workspace/
runs/<model_name>/<env_name>/<run_id>/checkpoints/H_002/workspace/
...
```

`H_000` 是初始 workspace。第 `t` 轮 agent 修改后的 workspace 保存为 `H_{t+1}`。

第一版 HLBench 使用 no-rollback 语义：无论本轮是否提升 validation，submitted workspace 都成为下一轮起点。`accepted` 是评估标签，不是状态推进条件。

## 输出文件

run-level artifacts：

```text
runs/<model_name>/<env_name>/<run_id>/events.jsonl
runs/<model_name>/<env_name>/<run_id>/transitions.jsonl
runs/<model_name>/<env_name>/<run_id>/learning_curve.json
runs/<model_name>/<env_name>/<run_id>/report/report.html
runs/<model_name>/<env_name>/<run_id>/report/learning_curve.svg
runs/<model_name>/<env_name>/<run_id>/report/metrics.json
```

per-epoch artifacts：

```text
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/events.jsonl
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/epoch.json
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/transition.json
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/evaluator/
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
