# Failure And Scoring

本文件定义 HLBench 中错误、失败、协议违规和最低分的处理方式。目标是让不同 agent、环境和 scenario 的结果可比较，避免崩溃或违规行为产生不可解释分数。

## Core Rule

HLBench 使用 fail-fast scoring：只要本轮出现执行层错误或协议违规，本轮 transition 直接给最低分。可以继续做诊断性评估和写 artifact，但最终分数使用 minimum score。

不算执行错误的情况：

- policy 正常执行但任务失败。
- episode 正常 timeout/truncation。
- environment return 较低。

这些属于策略表现差，应按环境实际 return 和 success 统计计分。

## Failure Taxonomy

### Policy-Level Failure

策略本身不可执行或运行时失败：

- `compile_error`: `system/policy.py` 语法错误。
- `import_error`: policy 模块导入失败。
- `missing_policy`: policy 文件、`Policy` 类或必要方法缺失。
- `runtime_exception`: `reset` 或 `act` 抛异常。
- `invalid_action`: 返回 action 不符合 `ActionSpec`。
- `timeout`: episode 达到 max steps 或环境 truncation，不属于执行错误。

处理规则：

- compile/import/missing policy: 本轮使用 minimum score。
- runtime exception: 本轮使用 minimum score。
- invalid action: 本轮使用 minimum score。
- timeout/truncation: 不触发 minimum score，按环境实际 return 计分，但 `success=false`。

### Agent-Level Failure

agent backend 执行失败，但不一定代表 policy 失效：

- agent 命令不存在。
- agent 超时。
- agent 非零退出码。
- final JSON 缺失或不可解析，当 `final_json_required=true` 时算 agent failure。
- stdout/stderr 超过限制。
- 没有修改 policy。

处理规则：

- 本轮使用 minimum score。
- 可以继续评估当前 workspace 中的 policy 作为诊断 artifact，但诊断结果不改变最低分。
- 记录 `agent_failed=true` 和具体 failure mode。
- 当 `final_json_required=false` 时，final JSON 不可解析只记录 metadata warning，不触发 minimum score。

### Contract Violation

agent 违反 workspace 或 benchmark 协议：

- 修改 `AGENTS.md`、`task.md`、`task_contract.json` 或 `feedback/`。
- 删除 `system/policy.py`。
- 写入 workspace 外或未授权目录。
- 尝试读取 validation / heldout 私有数据。
- 修改 evaluator、runner 或 protected files。
- 生成超大文件或不可接受 artifact。

处理规则：

- 标记 `invalid_transition=true` 并使用 minimum score。
- policy 可以继续评估，用于诊断，但最终 reward 不使用诊断分数。
- 篡改 task/feedback/evaluator 或读取私有数据属于严重违规，可进一步触发 run disqualification。
- 所有 violation 必须写入 `transition.json`。

## Minimum Score

默认最低分：

```json
{
  "minimum_score": {
    "total_reward": -1.0,
    "performance_score": 0.0,
    "invalid_transition": true,
    "reason": "execution_error_or_contract_violation"
  }
}
```

Recommended mapping:

| Condition | Scope | Rule |
| --- | --- | --- |
| compile error | transition | minimum score |
| import error | transition | minimum score |
| missing policy | transition | minimum score |
| runtime exception | transition | minimum score |
| invalid action | transition | minimum score |
| timeout/truncation | none | use environment return, success=false |
| agent timeout | transition | minimum score |
| agent non-zero exit | transition | minimum score |
| final JSON parse failure | transition | minimum score only if required |
| protected file modification | transition | minimum score |
| private data access attempt | run | minimum score plus possible disqualification |

Minimum score should be scenario configurable only when reward scale demands it. The default for normalized tasks is zero performance score and `total_reward=-1.0`.

## Reward Composition

When no minimum score is triggered, reward should separate performance from penalties:

```json
{
  "performance_score": 0.42,
  "train_delta": 0.10,
  "validation_delta": 0.05,
  "heldout_delta": 0.03,
  "complexity_penalty": -0.01,
  "total": 0.49
}
```

Rules:

- Performance should be computed from evaluator summaries.
- Agent execution failure triggers minimum score.
- Contract violation triggers minimum score.
- Invalid patch or policy execution failure triggers minimum score.
- Non-fatal metadata warnings may be recorded without changing score.

## Transition Fields

Every transition should include:

```json
{
  "result": {
    "policy_status": {
      "compile_ok": true,
      "import_ok": true,
      "missing_policy": false,
      "runtime_failures": []
    },
    "agent_status": {
      "backend": "command",
      "returncode": 0,
      "timed_out": false,
      "final_json_parse_ok": true,
      "agent_failed": false
    },
    "contract_status": {
      "invalid_transition": false,
      "violations": []
    },
    "minimum_score_applied": {
      "applied": false,
      "reason": null
    }
  }
}
```

Validation and heldout records must remain aggregate-only. Transition records must not include private seeds, private replays, private failure traces, or recoverable environment states.

## Open Decisions

- Whether minimum scores should be globally normalized to `[0, 1]` before reward deltas.
- Whether agent command count and train rollout count should be hard budgeted in the first implementation.
