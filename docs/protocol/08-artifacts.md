← [protocol index](./README.md)　|　← Previous: [§7 打分](./07-scoring.md)

# §8 产物布局

> 本章刻画 *一次 run 结束后落在磁盘的所有产物的位置、名字、文件 schema、跨文件不变量*。
>
> Workspace 内部（`system/` + `feedback/`）已在 [§1.2](./01-overview.md#12-三方角色与核心评测流程) / [§4](./04-feedback.md) 写过，本章只追加 workspace 在 run-level 的位置约定。

## 8.1 目录布局

```
runs/
└── <model>/
    └── <env>/
        └── <exp-id>/
            ├── run.json                  # 顶层元数据 + 最终分（本章 §8.4）
            ├── workspace/                # run 结束时的 workspace 快照
            │   ├── AGENTS.md
            │   ├── system/               # = checkpoints/submit_<best_submit_index>/
            │   └── feedback/             # submit_000/ ~ submit_NNN/ 全部
            ├── checkpoints/              # 每次 submit 的 system/ 快照
            │   ├── submit_000/
            │   │   └── metrics.json      # host-side static code metrics
            │   ├── submit_001/
            │   └── ...
            └── logs/                     # 实现者 & 分析者面向（agent 不可读）
                ├── harness.log
                ├── agent.jsonl
                └── env.log
```

三元组 `(model, env, exp_id)` 唯一定位一个 run。`runs/` 是协议规定 server 唯一在 workspace 之外写入的根目录；实现 **MAY** 允许通过 `HLBENCH_RUNS_DIR` 或 CLI flag 改路径，但 `runs/<model>/<env>/<exp_id>/` 以下的结构是 **normative**（不可改）。

## 8.2 ID 命名约定

### `model`

agent 身份 slug。**MUST** 匹配 `[a-z0-9][a-z0-9-]*`，最长 64 字符。

| 例 | 说明 |
|---|---|
| `claude-opus-4-7` | frontier LLM (claude backend) |
| `claude-code-sonnet` | claude backend default |
| `codex-gpt-5` | OpenAI codex backend |
| `reference-pd` | 参考实现（手写 PD） |

### `env`

env slug，**MUST** 与 env 注册名 byte-identical（同字段在 `/info:env`）。

| 例 | 说明 |
|---|---|
| `pendulum` | classic control |
| `halfcheetah` | MuJoCo |
| `minigrid_doorkey` | gridworld |

### `exp-id`

distinguishing 同一 `(model, env)` 下的多次 run。**SHOULD** 形如 `<purpose>-<YYYYMMDD>-<HHMM>`（便于排序与人读），**MUST** 匹配 `[a-zA-Z0-9._-]+`，最长 128 字符。

| 例 | 说明 |
|---|---|
| `v1paper-sonnet-b256-20260530-0154` | "paper v1 / sonnet / budget=256 / 时间戳" |
| `calibration-1` | 校准跑 |
| `eval-live-2` | 在线评测第 2 次 |

## 8.3 数字宽度（`submit_NNN` / `ep_<XXX>`）

`submit_NNN/` 与 `ep_<XXX>/` 都用零填充宽度：

```
width = max(3, len(str(episode_budget)))
```

| `episode_budget` | width | 例 |
|---|---|---|
| ≤ 999（典型） | 3 | `submit_000`、`submit_023`、`ep_142` |
| 1 000 – 9 999 | 4 | `submit_0000`、`submit_0234` |
| 10 000+ | 5+ | 按需 |

依据：每次 submit 至少消耗 1 episode，所以 submit 上界 = `episode_budget`，episode 上界亦同。固定宽度保证 lexicographic 排序无歧义。Width 在 run 开始时由 `/info:episode_budget` 推出，**run 内不变**。

`submit_NNN` 与 `ep_<XXX>` **是两个独立计数器**，宽度公式相同但数值无对齐关系（[§4.8](./04-feedback.md#48-episode-全局编号-xxx)）。

## 8.4 `run.json` schema

```json
{
  "schema_version": "0.1",
  "protocol_version": "protocol/v2.0-draft",

  "model": "claude-code-sonnet",
  "env": "pendulum",
  "exp_id": "v1paper-sonnet-b256-20260530-0154",

  "experiment_dimensions": {
    "episode_budget": 256,
    "min_episodes_per_submit": 1,
    "max_episodes_per_submit": 256,
    "seed_pool_id": "default",
    "agent_harness": "hlbench@0.1.0a1",
    "model_config": {"temperature": 1.0, "max_tokens": 8192}
  },

  "timing": {
    "start_time": "2026-05-29T17:54:48.356Z",
    "end_time": "2026-05-29T18:23:14.812Z",
    "wall_time_seconds": 1706.5
  },

  "outcome": {
    "status": "completed",
    "error": null,
    "final_score": 101.069,
    "best_submit_index": 7,
    "val_scores": {
      "0": -161.6,
      "1": -158.4,
      "3": -145.2,
      "5": -141.8,
      "7": -138.0,
      "9": -142.1
    },
    "heldout_mean_return": -138.775,
    "heldout_std_return": 12.34,
    "heldout_returns": [-105.0, -142.3, -98.0, "...256 floats total..."],
    "auxiliary": {
      "auc_in_loop": 94.73,
      "episodes_to_50pct": 4,
      "episodes_to_80pct": 16,
      "held_out_gap": -1.6,
      "val_heldout_gap": 0.775,
      "n_submits": 9,
      "n_successful_submits": 6,
      "episodes_used": 256,
      "mean_episodes_per_submit": 28.4,
      "mean_submit_wall_time": 189.6,
      "code_metrics_best": {
        "source_lines": 84,
        "functions": 5,
        "cyclomatic_total": 14,
        "tree_hash": "sha256:..."
      },
      "code_metrics_by_submit": {"7": {"source_lines": 84}},
      "code_metrics_trend": {
        "submits": [0, 3, 7],
        "source_lines": [21, 53, 84],
        "cyclomatic_total": [4, 9, 14]
      }
    }
  },

  "artifacts": {
    "workspace": "workspace/",
    "feedback": "workspace/feedback/",
    "checkpoints": "checkpoints/",
    "logs_harness": "logs/harness.log",
    "logs_agent": "logs/agent.jsonl",
    "logs_env": "logs/env.log"
  },

  "versions": {
    "harness": "0.1.0",
    "env": "0.1",
    "agents_md_hash": "sha256:8f3a...",
    "data_train_hash": "sha256:1a2b...",
    "data_valid_hash": "sha256:3c4d...",
    "data_heldout_hash": "sha256:5e6f..."
  }
}
```

### 字段说明（精选）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | string | 本 run.json 的 schema 版本（独立于 protocol_version） |
| `protocol_version` | string | 本 run 遵循的协议版本（如 `"protocol/v2.0"`），与 [§9](./09-versioning.md) 联动 |
| `outcome.status` | enum | `"completed"` / `"no_ok_submit"` / `"error"`（见下） |
| `outcome.error` | object \| null | `status == "error"` 时给 `{type, message, occurred_at_submit, traceback}`；其它 `null` |
| `outcome.final_score` | float \| null | 归一化分（[§7.2](./07-scoring.md#72-归一化公式)）；`status != "completed"` 时 `null` |
| `outcome.best_submit_index` | int \| null | finalize 时按 `val_score` 选出的 best；`no_ok_submit` 时 `null` |
| `outcome.val_scores` | object \| null | `submit_index → val_score` 字典；仅 `status == "completed"` 时给完整字典，否则 `null` |
| `outcome.heldout_returns` | float[] \| null | 256 个 heldout episode 的 raw return；用于离群点/方差分析 |
| `outcome.auxiliary.*` | object | [§7.5 Auxiliary metrics](./07-scoring.md#75-auxiliary-metrics) 全部 |
| `artifacts.*` | string | run 目录内的**相对路径**（**MUST**），便于 run 被移动 |
| `versions.harness` | string | hlbench 包版本（实现版） |
| `versions.env` | string | env 包版本 |
| `versions.agents_md_hash` | string | run 开始时 `AGENTS.md` 的 SHA-256 |
| `versions.data_train_hash` | string | run 开始时外部 `train.json` 的 SHA-256（有 data 目录时） |
| `versions.data_valid_hash` | string | run 开始时外部 `valid.json` 的 SHA-256（有 data 目录时） |
| `versions.data_heldout_hash` | string | run 开始时外部 `heldout.json` 的 SHA-256（有 data 目录时） |

### `outcome.status` 枚举

| 值 | 含义 | `final_score` |
|---|---|---|
| `completed` | run 走完正常 finalize | 实际分（含 `0` 时全部 ok 但归一化 ≤ 0） |
| `no_ok_submit` | agent 自始至终无 `status == ok` submit | `0` |
| `error` | harness crash / sandbox crash / 磁盘满 / etc. | `null`（`outcome.error` 描述原因） |

注：v1 的 `"aborted"`（agent / 操作者中途停）v2 **取消** —— agent 没有 `/finalize` 权限，操作者中途停归 `error`。

## 8.5 `workspace/` — 最终镜像

`runs/<...>/workspace/` 是 run 结束时 agent workspace 的快照，layout 同 [§1.2](./01-overview.md#12-三方角色与核心评测流程)：

```
workspace/
├── AGENTS.md
├── system/         # ⚠️ run 结束时 == checkpoints/submit_<best_submit_index>/
└── feedback/
    ├── submit_000/
    └── ...
```

### 不变量

**workspace/system/** 在 run 结束时 **MUST** byte-identical 于 `checkpoints/submit_<best_submit_index>/` 的 submitted code tree（排除 `_meta.json` / `metrics.json` 等 host metadata；v2 选择规则；v1 是 `final_submit_index = 最后一次 ok`，v2 已改）。

特殊情况：

- `outcome.status == "no_ok_submit"` → `workspace/system/` 是 run 结束时 agent 留下的最后内容（即使没 ok submit），不强制等于任何 checkpoint。
- `outcome.status == "error"` → workspace 可能 *不完整*（harness 来不及落最后状态）；消费者 **MUST** 容忍。

### Why keep it

`checkpoints/` 已保留每次 submit，但 `workspace/` 在顶层提供"最终 best 代码是哪个"的零跳转入口，便于工具直接取用。

## 8.6 `checkpoints/` — 历史代码快照

```
checkpoints/
├── submit_000/
│   ├── _meta.json          # 快照元信息（本节 §8.6.1）
│   ├── metrics.json        # host-side static code metrics（本节 §8.6.2）
│   ├── policy.py
│   ├── controllers/        # （如有）
│   ├── memory/             # （如有）
│   └── ...
├── submit_001/
└── ...
```

每个 `submit_NNN/` 是该次 submit 进入 Phase 2 Snapshot 时 `system/` 的**深拷贝**（排除 `__pycache__/` / `.git/` / 符号链接 / 缓存目录，见 [§5.3 Phase 2](./05-submit-lifecycle.md#phase-2--snapshot)）。

### `_meta.json` 字段

```json
{
  "schema_version": "0.1",
  "submit_index": 7,
  "submit_time": "2026-05-29T18:15:32Z",
  "n_episodes_requested": 32,
  "remaining_budget_before": 96,
  "remaining_budget_after": 64,
  "snapshot_size_bytes": 4128,
  "snapshot_files": ["policy.py", "controllers/pid.py", "memory/stats.json"],
  "import_scan": ["numpy", "math", "controllers.pid"],
  "validation_status": "ok"
}
```

`validation_status` 用 [§5.5 verdict 10 枚举](./05-submit-lifecycle.md#55-verdict-完整枚举10-个) 同一套，与 `workspace/feedback/submit_NNN/summary.json:status` byte-identical。

### `metrics.json` 字段

`metrics.json` 由 server 在 snapshot 后自动计算，agent run 期间不可见，且不参与 best 选择或 final score。它用于事后分析软件复杂度和工程性。字段包括：

```json
{
  "schema_version": "0.1",
  "files": 4,
  "python_files": 3,
  "bytes": 4128,
  "policy_bytes": 2048,
  "source_lines": 84,
  "classes": 1,
  "functions": 5,
  "imports": ["math", "statistics"],
  "cyclomatic_total": 14,
  "cyclomatic_max": 5,
  "parse_errors": [],
  "test_files": 1,
  "tree_hash": "sha256:..."
}
```

`_meta.json` 与 `metrics.json` 是 checkpoint metadata；run 结束 mirror 到 `workspace/system/` 时 **MUST NOT** 复制这些 host metadata。

### 失败 submit 也保留快照

即使 `validation_status != "ok"`，checkpoint 目录仍然创建——agent 要能复盘"我那次失败的代码长啥样"。

### 存储

实现 **MAY** 用 hardlink / content-addressed 去重（多个 submit 共享相同文件），只要消费者看见的目录视图与原始拷贝一致即可。

## 8.7 `logs/` — 实现者 & 分析者面向

> **`logs/` 对 agent 完全不可见**。诊断信息要给 agent 的 **MUST** 走 `workspace/feedback/`（[§4](./04-feedback.md)），**禁止**把 agent 需要的信息放 `logs/` 作捷径。

### `harness.log`（JSONL，每行一 framework 事件）

```jsonl
{"timestamp":"2026-05-29T17:54:48.356Z","event":"run.open","model":"codex","env":"cartpole","budget":16}
{"timestamp":"2026-05-29T17:54:49.012Z","event":"submit.start","submit_index":0,"cases":[0,1,2,3]}
{"timestamp":"2026-05-29T17:54:49.245Z","event":"submit.snapshot","submit_index":0,"checkpoint":"checkpoints/submit_000"}
{"timestamp":"2026-05-29T17:54:49.890Z","event":"submit.feedback","submit_index":0,"status":"ok","mean_return":500.0}
{"timestamp":"2026-05-29T18:22:30.123Z","event":"run.auto_close.start","n_snaps":4}
{"timestamp":"2026-05-29T18:23:14.812Z","event":"run.close","outcome":"completed","best_submit_index":3}
```

The exact event set is implementation-specific, but local EvoPolicyGym runs write
run, server, loop, submit, checkpoint, feedback, validation/final eval,
workspace mirror, and close events here.

**MUST NOT** 出现 hidden seed 真值、`expert_baseline` / `random_baseline` 数值。

### `agent.jsonl`（JSONL，每行一 agent 事件）

```jsonl
{"t":"2026-05-29T17:54:48.500Z","event":"agent_start","model":"claude-code-sonnet"}
{"t":"2026-05-29T17:54:51.123Z","event":"completion","input_tokens":12500,"output_tokens":847,"latency_ms":3200,"cost_usd":0.184}
{"t":"2026-05-29T17:54:51.140Z","event":"tool_call","tool":"http_get","args":{"path":"/task"}}
{"t":"2026-05-29T17:54:55.200Z","event":"submit","n_episodes":32}
{"t":"2026-05-29T18:22:00.000Z","event":"agent_end","reason":"budget_exhausted"}
```

便于事后审计 token / cost / 决策时序。冗长 chain-of-thought **MAY** 截断，但 **MUST** 标 `"truncated": true`。

### `env.log`（plain text，常空）

env 端非致命警告（MuJoCo physics warning、deprecation notice 等）。失败 / OOM / timeout 等 **不**放这里——它们走 `workspace/feedback/.../errors.txt`。

### 压缩

logs **MAY** 用 `.gz` 压缩（`harness.log.gz` / `agent.jsonl.gz`）；消费者 **MUST** 透明处理两种扩展。

## 8.8 多变量实验

跑 sweep（不同 budget / 不同 seed pool / 不同 agent config）时，**把变量编进 `exp_id`，不要加层级**：

```
runs/claude-code-sonnet/pendulum/
├── b64__s42__20260529-1900__abc/
├── b128__s42__20260529-2000__def/
├── b256__s42__20260530-0100__ghi/
└── b256__s43__20260530-0200__jkl/
```

约定前缀（建议、**不强制**）：

| 前缀 | 含义 |
|---|---|
| `b<N>` | `episode_budget = N` |
| `s<N>` | `seed_pool_id` offset |
| `t<N>` | temperature × 10（`t10` = 1.0） |

**`experiment_dimensions`** 在 `run.json` 里是 sweep 参数的 **authoritative** 记录；exp_id 前缀只是人读便利，分析工具应直接读 `experiment_dimensions`。

## 8.9 跨文件不变量

`hlbench check`（run 健康度）**MUST** 验证：

| # | 不变量 | 范围 |
|---|---|---|
| R1 | `run.json:experiment_dimensions.episode_budget` = server 实际 `episode_budget` | per run |
| R2 | `outcome.auxiliary.episodes_used` = `Σ summary.json:n_episodes`（含执行级失败） | per run |
| R3 | `outcome.auxiliary.n_submits` = `count(workspace/feedback/submit_*/)`（含失败） | per run |
| R4 | `outcome.auxiliary.n_successful_submits` = `count(... where summary:status == "ok")` | per run |
| R5 | `versions.env` = server 实际 `env_version` | per run |
| R6 | `env` 字段 = server 实际 `env` | per run |
| R7 | `outcome.best_submit_index` ∈ `outcome.val_scores` 键集合 | `status == "completed"` |
| R8 | `outcome.val_scores` 键集合 = `{NNN | workspace/feedback/submit_NNN/summary.json:status == "ok"}` | `status == "completed"` |
| R9 | `workspace/system/` byte-identical 于 `checkpoints/submit_<best_submit_index>/`（排除 `_meta.json` / `metrics.json`） | `status == "completed"` |
| R10 | `len(outcome.heldout_returns) == 256` | `status == "completed"` |
| R11 | `outcome.heldout_returns` 内绝无 raw seed 值 | per run |
| R12 | `artifacts.*` 全部是 **相对路径**（无 `/` 开头、无 `..` 上行） | per run |
| R13 | 每个 accepted submit checkpoint 都有 `metrics.json`，且与 checkpoint code tree 重新计算结果一致 | per run |
| R14 | `outcome.auxiliary.code_metrics_best/by_submit/trend` 与 checkpoint `metrics.json` 一致 | per run |

R7 + R8 是 v2 新增的 val-based selection 一致性保证：`best_submit_index` 必须来自 val_scores 字典的键集合，而 val_scores 字典的键又必须恰好覆盖所有 ok submit。

## 8.10 实现者不变量

harness **MUST** 保证：

1. **`run.json` atomic write**：temp file + rename，永远观察不到半写。
2. **Checkpoint 不可变**：`checkpoints/submit_NNN/` 一旦创建，内容不再变动。
3. **Workspace 镜像**：run 结束时 `workspace/system/` 与 best checkpoint submitted code tree byte-identical（R9）。
4. **Logs append-only**：`logs/*.{log,jsonl}` 严禁 rewrite history。
5. **无 hidden seed 泄漏**：hidden seeds 与 per-episode heldout returns 仅以已聚合统计形式出现在 `run.json`，**不出现**在 `logs/` / `workspace/feedback/` / `workspace/`。
6. **Path stability**：`run.json:artifacts` 全部相对路径（R12）。
7. **Schema versioning**：每个 JSON schema 文件（`run.json` / `summary.json` / `_meta.json` / `metrics.json`）和 `/info` 响应都带 `schema_version`；harness 写自身实现的版本，消费者 **SHOULD** 解析前校验。

## 8.11 复现性

bit-exact 复现一次 run 需要：

- 同 `versions.harness`
- 同 `versions.env` 与同 `versions.data_*_hash`（三池外部数据 byte-identical）
- 同 `versions.agents_md_hash`
- 同 agent harness + model
- 同 `experiment_dimensions`

但 **LLM-driven run 本质非确定**（temperature、server-side 浮点路径）。本协议**不要求**跨 run bit-exact；要求的是：**给定相同的 best ckpt 与 same env_version，heldout 评估出来的 `final_score` 必须 bit-exact**（即评分链路本身确定）。

agent 行为的非确定性以 **跨多 `exp_id` 报方差** 的方式承认与披露，不在协议层"消除"。

---

← Previous: [§7 打分](./07-scoring.md)　|　Next: [§9 版本管理](./09-versioning.md) →
