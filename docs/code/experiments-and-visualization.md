# Human Analysis And Visualization

本文定义 HLBench 面向 benchmark 维护者、研究者和报告读者的实验分析与数据可视化边界。它不是 agent-facing 协议：报告、图表、`metrics.json` 和跨 split 分析结果不得复制进 learner workspace，也不得作为下一轮 agent prompt 或反馈。

核心原则：

- agent 在 workspace 内只能看到 task contract、public train feedback 和自己创建的 train-only artifacts。
- benchmark 在 workspace 外执行 standard evaluation 并生成 human-facing analysis。
- human-facing report 只能展示允许公开的聚合数据。
- validation / heldout 的可视化必须保持 aggregate-only；private evaluator 不得生成 seed、replay、per-episode record 或 failure trace artifacts。

## Experiment Types

### Standard Evaluation

由 benchmark runner 触发，作为正式评分依据。

```text
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/
  input/policy.py
  submission/policy.py
  evaluation/train/
  evaluation/validation/
  evaluation/heldout/
```

规则：

- train 可以保存 summary、episodes、failures 和 replay。
- validation / heldout 只生成并保存 aggregate summary；不得生成 replay、trace、episodes 或 failure-detail files。
- standard evaluation 结果进入 `transition.json`、`learning_curve.json` 和 final report。

### Agent-Created Train Experiment

由 agent 在 learner workspace 内主动触发，只允许使用 train split。本文讨论它们是因为 human-facing report 可以在 run 结束后汇总这些 artifacts；这不表示 report 会回流给 agent。

```text
workspace/experiments/
  notes.md
  probe_key_logic/
    experiment.json
    summary.json
    failures.jsonl
    replays/
```

规则：

- agent experiment 可以读取 workspace 中公开的 aggregate validation summary，但不能访问 validation seed、replay、per-episode record 或 failure details；heldout 完全不可见。
- agent-created train experiment 不直接决定最终分数。
- agent-created train experiment 可以作为诊断 artifact 保留在 checkpoint 中。
- benchmark 可以统计 agent-created train experiment 的 rollout cost，但不把它当作 private evaluation。
- final report 可以引用这些 train-only artifacts，但 report 本身不进入 workspace。

## Experiment Manifest

每个实验目录应该包含 `experiment.json`：

```json
{
  "experiment_id": "probe_key_logic",
  "source": "agent",
  "epoch": 0,
  "split": "train",
  "policy_sha256": "...",
  "command": "python -m hlbench.rollout --workspace . --split train --episodes 5 --output-dir experiments/probe_key_logic",
  "episodes": 5,
  "started_at": "2026-05-15T00:00:00Z",
  "ended_at": "2026-05-15T00:00:10Z",
  "artifacts": {
    "summary": "summary.json",
    "failures": "failures.jsonl",
    "replays": "replays/"
  }
}
```

`source` values:

- `benchmark`: official evaluator run.
- `agent`: train-only experiment created inside workspace.
- `diagnostic`: optional runner-created diagnostic after minimum score, not used for scoring.

## Metrics Tables

The human analysis pipeline should build normalized tables from transitions and summaries.

### `metrics.json`

```json
{
  "run_id": "example",
  "model_name": "claude-code",
  "env_name": "cartpole_balance",
  "scenario_id": "cartpole_balance_v0",
  "epochs": 5,
  "primary": {
    "heldout_return_auc": 0.72,
    "final_heldout_mean_return": 0.91,
    "best_heldout_mean_return": 0.94
  },
  "quality": {
    "invalid_transition_rate": 0.2,
    "minimum_score_count": 1,
    "agent_failure_count": 0,
    "contract_violation_count": 1
  },
  "cost": {
    "train_episodes": 120,
    "agent_wall_time_seconds": 640.5
  }
}
```

### `learning_curve.json`

One row per epoch/checkpoint:

```json
{
  "epoch": 0,
  "checkpoint": "checkpoints/H_001/workspace",
  "train_mean_return": 0.4,
  "validation_mean_return": 0.35,
  "heldout_mean_return": 0.33,
  "train_success_rate": 0.5,
  "validation_success_rate": 0.4,
  "heldout_success_rate": 0.4,
  "minimum_score_applied": false,
  "invalid_transition": false,
  "policy_sha256": "..."
}
```

Validation and heldout rows must remain aggregate-only.

## Visualization Requirements

The final human-facing report should prioritize plots that explain learning behavior:

- heldout mean return over epochs, with minimum score markers;
- train / validation / heldout aggregate curves;
- success rate over epochs;
- mean steps over epochs;
- invalid transition and minimum score timeline;
- policy complexity growth;
- rollout cost and wall-clock cost;
- per-epoch transition table with links to artifacts.

Optional train-only visualizations for human analysis:

- train failure mode counts;
- train replay viewer;
- train action histogram;
- agent-created train experiment comparison table.

禁止展示：

- validation / heldout replay;
- validation / heldout per-episode records;
- validation / heldout seeds;
- private failure traces;
- recoverable private environment states.

这些 private artifacts 不只是禁止展示，也禁止由 evaluator 生成。

## Report Layout

Recommended output:

```text
runs/<model_name>/<env_name>/<run_id>/report/
  index.html
  metrics.json
  learning_curve.json
  learning_curve.svg
  success_curve.svg
  complexity.svg
  transitions.html
```

`index.html` should be static and self-contained where practical. It should link to transition JSON, submission metadata, and train-only artifacts.

Report files live under `runs/<model_name>/<env_name>/<run_id>/report/`. They are not copied into `workspace/`, `feedback/`, or any checkpoint workspace.

## Data Flow

```text
transition.json
transitions.jsonl
learning_curve.json
evaluator summaries
agent run metadata
complexity metrics
  -> report metrics builder
  -> metrics.json
  -> static plots
  -> report/index.html
```

The report builder must not read learner workspace as an agent input source. It should consume run-dir artifacts and checkpoint snapshots after the fact, apply the public/private data boundary, and render human-facing analysis only.

## Minimum Score Visualization

Minimum score should be visible in human-facing charts and tables:

```json
{
  "epoch": 2,
  "minimum_score_applied": true,
  "reason": "invalid_action",
  "invalid_transition": true
}
```

Plots should mark such epochs with a distinct point marker. Tables should show the reason and link to the relevant transition.

## Open Decisions

- Whether agent experiments should count against a hard rollout budget in v1.
- Whether train replay visualization is required or optional.
- Whether reports should show validation curves by default or behind an advanced/debug section.
