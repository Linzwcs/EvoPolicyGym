# Workspace 契约

HLBench 的 workspace 是 benchmark 暴露给 learner 的公共操作界面。第一版应该保持足够简单，避免把原型实现细节或过早的系统设计塞进 agent 工作区。

一个正常的 HL workspace 只需要回答四个问题：

- 任务是什么；
- 当前 policy 实现在哪里；
- 当前可见的训练反馈在哪里；
- agent 可以把自己的分析工具和临时实验结果放在哪里。

## MVP 目录结构

推荐第一版使用：

```text
workspace/
  AGENTS.md
  task.md

  system/
    policy.py

  feedback/
    current/
      summary.json
      episodes.jsonl
      failures.jsonl
      replays/
    history/
      epoch_000/
        train/
          summary.json
          episodes.jsonl
          failures.jsonl
          replays/
        validation_summary.json
        manifest.json
      epoch_001/
        train/
          summary.json
          episodes.jsonl
          failures.jsonl
          replays/
        validation_summary.json
        manifest.json

  tools/

  experiments/
```

其中：

- `AGENTS.md` 是 agent 规则，只读；
- `task.md` 是任务说明，只读；
- `system/` 是核心 heuristic system，可编辑；
- `feedback/` 是 benchmark 生成的 train feedback 和历史 aggregate validation score，只读；
- `tools/` 初始可以为空，agent 可以写自己的分析脚本；
- `experiments/` 保存 agent 主动运行 train rollout 产生的结果和临时分析输出。

不建议第一版放 `metadata/`。runner 状态、私有评估结果、validation/heldout seed、event trace 和 checkpoint 索引都应该在 workspace 外部，由 benchmark run directory 管理。validation aggregate 可以通过 `feedback/history/` 暴露给 agent；heldout 不进入 workspace。

## `AGENTS.md`: 执行约束

`AGENTS.md` 面向 coding agent。它定义工作规则，而不是解释任务内容。

应该包含：

```text
Editable paths:
- system/
- tools/
- experiments/

Read-only paths:
- AGENTS.md
- task.md
- feedback/

Allowed feedback:
- latest train feedback
- historical train feedback
- aggregate validation summaries in feedback/history

Rules:
- Do not modify task.md or feedback/.
- Do not attempt to inspect validation seeds, validation replay, heldout data, or private evaluator artifacts.
- Keep system/policy.py executable.
- Use tools/ only for analysis helpers, not for changing the evaluator.
- Put temporary rollout outputs and notes under experiments/.
```

`AGENTS.md` 应该由 benchmark 生成并保持只读。agent 修改它应算 violation。

## `task.md`: 任务说明

`task.md` 是单个 Markdown 文件，描述环境和任务契约。

它应该包含：

- 环境名称；
- 游戏目标；
- observation 格式；
- action 空间；
- reward 含义；
- success 条件；
- episode 限制；
- policy 接口；
- 允许的 train rollout 命令；
- 公开反馈文件说明；
- 常见失败模式。

示例结构：

```text
# Task

Environment: MiniGrid-KeyCorridorS3R2-v0

Goal:
  Pick up the correct key, open the matching door, and reach the goal.

Observation:
  ...

Actions:
  0 = left
  1 = right
  ...

Scoring:
  The main score is the environment episode return.

Policy interface:
  Implement system/policy.py.

Train rollout:
  python -m hlbench.rollout --workspace . --split train --episodes 10 --output-dir experiments/<name>
```

第一版不需要 `task/observation.md`、`task/actions.md`、`task/scoring.md` 这些拆分文件。复杂任务未来可以引入附件，但默认契约应保持 `task.md` 单文件。

## `system/`: 核心 Policy 实现

`system/` 是 agent 要持续优化的 heuristic system。

第一版建议只放代码，不放 Markdown：

```text
system/
  policy.py
  helpers.py        # optional
  __init__.py       # optional
```

`system/policy.py` 是必须文件。其他 Python 文件可以由 agent 添加，用于组织复杂策略逻辑。

推荐 policy 接口：

```python
class Policy:
    def reset(self, seed=None):
        pass

    def act(self, observation, reward, terminated, truncated, info):
        return action
```

如果具体环境使用函数式接口，也可以是：

```python
def act(observation, reward, terminated, truncated, info):
    return action
```

接口必须在 `task.md` 中明确。

不建议把 `memory.md`、`regressions.md` 放进 `system/`。如果 agent 需要记录想法或回归清单，可以放在 `experiments/notes.md` 或 `tools/` 中的分析脚本里。`system/` 应尽量保持为可执行策略实现。

## `feedback/`: 环境生成的公开反馈

`feedback/` 由 benchmark 生成，agent 只读。

它包含 train split 的公开反馈，以及历史 aggregate validation score。不包含 validation replay、validation per-episode records、validation failure details、seed 或任何 heldout 信息。

推荐结构：

```text
feedback/current/
  summary.json
  episodes.jsonl
  failures.jsonl
  replays/

feedback/history/
  epoch_000/
    train/
      summary.json
      episodes.jsonl
      failures.jsonl
      replays/
    validation_summary.json
    manifest.json
  epoch_001/
    train/
      summary.json
      episodes.jsonl
      failures.jsonl
      replays/
    validation_summary.json
    manifest.json
```

### `feedback/current/`

当前 epoch 开始时，`feedback/current/` 来自上一轮 submission 的公开 train feedback。epoch 0 没有初始 policy 预运行，因此只写一个 empty marker；agent 先修改 policy，再由 benchmark 评估 submission，并在评估后把最新 train feedback 写回 `feedback/current/`。

典型文件：

- `summary.json`: mean return、success rate、mean steps、failure mode 计数；
- `episodes.jsonl`: 每个 train episode 的结果；
- `failures.jsonl`: 失败 episode 的摘要；
- `replays/`: 可选，train replay 或 trace。

### `feedback/history/`

历史 epoch 的 agent-visible feedback。它帮助 agent 比较学习过程：

- `train/` 可以包含旧的 train summary、episodes、failures 和 replays；
- `validation_summary.json` 只包含 aggregate validation 分数；
- 不包含 validation replay、validation per-episode records、validation failures、seed 或 heldout 信息。

命名按 epoch，而不是按某次临时 rollout：

```text
feedback/history/epoch_003/
```

这样含义更清楚：这是第 3 轮外层学习结束后由 benchmark 记录的标准 train feedback 和 aggregate validation summary。

## `tools/`: Agent 编写的分析工具

`tools/` 初始可以为空。

它不是 benchmark 提供 evaluator 的地方，而是 agent 可以写分析脚本的地方。例如：

```text
tools/
  inspect_failures.py
  compare_episodes.py
  visualize_trace.py
```

这些脚本可以读取：

- `task.md`
- `system/`
- `feedback/`
- `experiments/`

但不应该依赖 validation/heldout，也不能修改 benchmark evaluator。

官方 train rollout 命令不一定要放在 `tools/`。更干净的方式是由 benchmark 提供外部 CLI，并在 `task.md` 中说明：

```text
python -m hlbench.rollout --workspace . --split train --episodes 10 --output-dir experiments/<name>
```

这样可以避免 agent 修改 rollout runner。

## `experiments/`: Agent 的临时实验区

`experiments/` 是 agent 可写目录，用于保存它主动运行的 train rollout、临时分析结果和 notes。

示例：

```text
experiments/
  notes.md
  rollout_after_key_logic/
    summary.json
    episodes.jsonl
    failures.jsonl
  compare_wall_collisions.json
```

如果 agent 调用官方 train rollout CLI，输出应写入 `experiments/<agent_chosen_name>/`。

`experiments/` 可以随 workspace checkpoint 一起保存，因为它是 agent 可见学习历史的一部分。但训练数据构建时，可以只抽取摘要，避免样本过大。

## 不放进 Workspace 的东西

以下内容不应放入 agent workspace：

- validation seed list；
- heldout seed list；
- validation / heldout failure details；
- evaluator source code；
- reward implementation internals；
- private run metadata；
- checkpoint index；
- event trace；
- final report；
- hidden simulator state；
- solution policy。

这些内容由 benchmark run directory 管理，例如：

```text
runs/<run_id>/events.jsonl
runs/<run_id>/transitions.jsonl
runs/<run_id>/epochs/
runs/<run_id>/checkpoints/
runs/<run_id>/report/
```

它们是审计和报告材料，不是 learner 的可见输入。

## 可编辑与只读边界

推荐可编辑：

```text
system/
tools/
experiments/
```

推荐只读：

```text
AGENTS.md
task.md
feedback/
```

runner 应该在 agent 运行前后检查只读路径 hash。只读路径被修改时，该 epoch 应标记为 violation。

## 为什么这样更合适

这个布局把角色分清楚：

- `task.md` 是任务契约；
- `system/` 是要优化的 policy code；
- `feedback/` 是环境给出的不可修改训练证据；
- `tools/` 是 agent 自己写的分析能力；
- `experiments/` 是 agent 自己产生的探索历史；
- runner 私有状态留在 workspace 外部。

这比把 `memory.md`、`regressions.md`、`metadata/`、benchmark tools 全部预塞进 workspace 更干净，也更适合迁移到新仓库重构。
