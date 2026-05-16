# Visibility And Data Boundary

本文定义 HLBench 数据可见性边界。核心目标是把 agent-facing workspace、benchmark-private evaluation 和 human-facing analysis 分开。validation 只允许 aggregate score 进入 history；heldout 完全不进入 workspace。

## Visibility Classes

### Agent-Facing

agent 可以在 learner workspace 中读取的数据：

- `AGENTS.md`
- `task.md`
- `task_contract.json`
- `system/`
- `feedback/current/`
- `feedback/history/`
- `tools/`
- `experiments/`

只能包含 public train 信息和 agent 自己创建的 train-only artifacts。

### Benchmark-Private

只允许 runner/evaluator 使用，不复制进 workspace：

- validation / heldout seed split
- private evaluator config
- heldout aggregate summaries before report rendering
- checkpoint index
- event stream
- raw transition artifacts before filtering

### Human-Facing

run 结束后给维护者和研究者看的分析结果：

- `runs/<model_name>/<env_name>/<run_id>/report/`
- `metrics.json`
- aggregate learning curves
- static plots
- transition table

Human-facing artifacts 不得复制进 workspace，也不得作为下一轮 agent prompt 或 feedback。

## Data Placement Matrix

| Data | Workspace | Run Dir | Human Report | Forbidden |
| --- | --- | --- | --- | --- |
| task contract | yes | yes | optional | no |
| policy code | yes | yes/checkpoint | link/diff | no |
| train summary | yes | yes | yes | no |
| train failures | yes | yes | optional | no |
| train replay | yes | yes | optional train-only | no |
| train per-episode records | yes | yes | optional train-only | no |
| validation summary | aggregate history only | yes | yes aggregate | no |
| heldout summary | no | yes | yes aggregate | no |
| validation seeds | no | private only | no | workspace/report |
| heldout seeds | no | private only | no | workspace/report |
| validation replay | no | no | no | must not be generated |
| heldout replay | no | no | no | must not be generated |
| validation per-episode records | no | no | no | must not be generated |
| heldout per-episode records | no | no | no | must not be generated |
| validation failure details | no | no | no | must not be generated |
| heldout failure details | no | no | no | must not be generated |
| agent stdout/stderr | no | submission files | link/metadata | workspace |
| report metrics | no | yes | yes | workspace |
| report plots | no | yes | yes | workspace |

## Allowed Workspace Feedback

`workspace/feedback/` may contain:

```text
feedback/current/
  summary.json
  episodes.jsonl
  failures.jsonl
  replays/

feedback/history/epoch_000/
  train/
    summary.json
    episodes.jsonl
    failures.jsonl
    replays/
  validation_summary.json
  manifest.json
```

Train files may include replays and per-episode records. Validation history is
aggregate-only and must not include seeds, replay paths, per-episode records,
failures, actions, observations, or recoverable environment states. Heldout
must never appear in workspace feedback. If heldout data or validation details
appear in workspace feedback, the transition is invalid and minimum score applies.

## Private Evaluation Rules

Validation and heldout evaluation may write only aggregate summaries to run-dir artifacts:

```json
{
  "split": "heldout",
  "episodes": 50,
  "mean_return": 0.93,
  "success_rate": 1.0,
  "mean_steps": 20.6
}
```

It must not include:

- seed values;
- per-episode records;
- action sequences;
- observation sequences;
- replay paths;
- failure mode details;
- recoverable environment states.

Private evaluators must not create replay, trace, per-episode record, action sequence, observation sequence, or failure-detail files for validation / heldout, even temporarily. The implementation should disable those writers before rollout starts, not generate and delete them later. Aggregate validation summaries may be copied into `feedback/history/`; heldout summaries remain run-dir/report-only.

## Evaluation Directory Layout

Each epoch separates public train artifacts from aggregate-only validation / heldout artifacts:

```text
runs/<model_name>/<env_name>/<run_id>/epochs/epoch_000/
  input/
    policy.py
  submission/
    policy.py
    agent.json
    compile.json
  evaluation/
    train/
    validation/
    heldout/
```

`evaluation/train/` is the public train feedback source copied into
`workspace/feedback/current/` and `workspace/feedback/history/`. It may contain
train episode records, failures, and replays. `evaluation/validation` and
`evaluation/heldout` may contain only `summary.json` and `manifest.json`.

## Transition And Learning Curve

`transition.json`, `transitions.jsonl`, and `learning_curve.json` may contain aggregate validation / heldout metrics, but not private details.

Allowed:

```json
{
  "heldout_mean_return": 0.93,
  "heldout_success_rate": 1.0,
  "heldout_mean_steps": 20.6
}
```

Forbidden:

```json
{
  "heldout_seed_pool": "default/heldout",
  "heldout_failures": [],
  "heldout_replays": "replays/"
}
```

## Checkpoint Rules

Checkpoints save learner workspace state:

```text
runs/<model_name>/<env_name>/<run_id>/checkpoints/H_001/workspace/
```

Checkpoint workspaces must not contain:

- human-facing report files;
- heldout summaries;
- validation / heldout seeds;
- validation replay, per-episode records, or failure details;
- private evaluator logs;
- report metrics or plots.

If a checkpoint contains private data, the run should be marked invalid because future epochs could read it.

## Human-Facing Report Rules

`runs/<model_name>/<env_name>/<run_id>/report/` may contain aggregate validation / heldout curves and tables. It may link to train-only artifacts. It must not contain private per-episode data.

Reports are for humans after or outside agent execution. They are not agent feedback.

## Violation Examples

- `task_contract.json` includes heldout seed values.
- `workspace/feedback/current/summary.json` comes from validation split.
- `workspace/experiments/debug/` contains heldout replay.
- validation evaluator generates replay files, even if they are later deleted.
- `transition.json` stores validation action sequences.
- `report/` is copied into `workspace/`.
- `checkpoints/H_002/workspace/` contains `metrics.json` from a previous report.

Any such violation should mark the transition invalid and apply minimum score. Private data access attempts may also disqualify the whole run.
