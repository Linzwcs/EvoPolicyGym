# 复杂度评估

HLBench 需要引入传统软件工程复杂度指标，因为 heuristic learning 很容易出现一种伪提升：

```text
分数小幅提高，但 policy 代码持续膨胀、分支越来越多、规则越来越脆弱。
```

这类系统可能在当前 train seeds 上有效，但难以维护、难以继续优化，也更容易产生 regression。复杂度评估的目的不是鼓励最短代码，而是衡量 agent 是否能在持续改进中保持 heuristic system 可维护。

## 评估范围

复杂度只评估 workspace 中的核心策略实现：

```text
workspace/system/
```

第一版主要评估：

```text
workspace/system/policy.py
workspace/system/*.py
```

不评估：

- `feedback/`
- `experiments/`
- `tools/`
- `task.md`
- `AGENTS.md`
- benchmark runner 代码
- evaluator 代码

原因是 `tools/` 和 `experiments/` 是 agent 的分析过程产物。它们可以很大，但不一定进入最终 policy。HLBench 要惩罚的是被持续携带到 `H_t` 中的 policy complexity。

## 指标分层

### 1. Size Metrics

衡量代码规模：

```text
sloc
logical_loc
file_count
function_count
class_count
comment_lines
blank_lines
patch_added_lines
patch_deleted_lines
```

这些指标简单但有用。很多失败的 heuristic learning 会表现为：

```text
return 不变，但 sloc 持续增加
```

这应被视为无收益复杂度增长。

### 2. Structural Complexity

衡量控制流复杂度：

```text
cyclomatic_complexity_mean
cyclomatic_complexity_max
branch_count
loop_count
max_nesting_depth
return_count
exception_handler_count
```

最重要的是 cyclomatic complexity 和 max nesting depth。

高分 policy 不一定要低复杂度，但如果单个函数变成一个巨大条件树，后续 agent 很难安全修改它。HLBench 应该把这种增长记录出来。

### 3. Maintainability Metrics

可以使用传统指标：

```text
maintainability_index
halstead_volume
halstead_difficulty
halstead_effort
```

这些指标不应该单独决定胜负，但适合作为复杂度报告和 penalty 的组成部分。

Python 第一版可以考虑用 `radon` 计算：

```text
radon raw
radon cc
radon mi
radon hal
```

如果不想引入外部依赖，也可以先用 AST 自己计算 size、branch、nesting、function length。

### 4. Coupling And Surface Area

衡量 policy 是否变得难以隔离：

```text
import_count
external_dependency_count
module_count
public_function_count
global_mutable_state_count
large_literal_table_size
```

第一版应尤其关注：

- 是否引入非允许依赖；
- 是否大量使用全局可变状态；
- 是否用巨大 hard-coded table 过拟合训练 seeds；
- 是否把 policy 拆成过多文件导致审计成本上升。

### 5. Duplication

衡量重复逻辑：

```text
duplicate_line_blocks
similar_branch_count
repeated_literal_count
```

MVP 可以先不做复杂 clone detection，只记录简单信号：

- 相同连续代码块；
- 重复 magic constants；
- 多个结构相似的 `if` 分支。

## 绝对复杂度与增量复杂度

每个 checkpoint 都应记录绝对复杂度：

```json
{
  "checkpoint": "H_004",
  "sloc": 184,
  "function_count": 12,
  "cyclomatic_complexity_max": 18,
  "max_nesting_depth": 5,
  "maintainability_index": 61.2
}
```

每个 transition 还应记录增量：

```json
{
  "epoch": 4,
  "complexity_before": {
    "sloc": 160,
    "cyclomatic_complexity_max": 14
  },
  "complexity_after": {
    "sloc": 184,
    "cyclomatic_complexity_max": 18
  },
  "complexity_delta": {
    "sloc": 24,
    "cyclomatic_complexity_max": 4
  }
}
```

HLBench 更关心增量，因为它评测的是系统演化过程。

## Complexity Penalty

复杂度 penalty 不应该简单惩罚所有新增代码。很多真实改进需要引入状态解析、路径规划或 recovery logic。

更合理的是惩罚无收益增长：

```text
if validation_delta <= epsilon:
  complexity_penalty = alpha * positive_complexity_delta
else:
  complexity_penalty = beta * max(0, positive_complexity_delta - free_budget)
```

其中：

- `alpha > beta`；
- 没有带来分数提升的复杂度增长应被更强惩罚；
- 带来稳定提升的复杂度增长可以有少量 free budget；
- 如果复杂度下降且分数不降，应该记录为 compression improvement。

也可以报告复杂度效率：

```text
complexity_efficiency = validation_delta / max(1, added_sloc)
```

它用于解释一个 agent 是否用很小的代码增量获得较大改进。

## Compression Improvement

HL 的长期能力不只是吸收失败，还包括压缩历史。

好的 agent 应该能做到：

```text
score 不下降，同时删除过时分支、合并重复逻辑、降低 nesting、减少特殊 case。
```

因此报告中应记录：

```text
compression_epochs
```

定义可以是：

```text
validation_delta >= -epsilon
and complexity_delta < 0
```

这类 epoch 对 HLBench 很重要，因为它说明 agent 不只是在堆规则。

## Hard Limits 与 Soft Penalties

复杂度指标有两种用途。

### Hard Limits

用于防止明显异常：

- `system/` 文件数量超过上限；
- `system/` 总行数超过上限；
- 单文件超过上限；
- 单函数 cyclomatic complexity 超过极高阈值；
- 引入禁止依赖；
- 生成巨大 hard-coded lookup table。

触发 hard limit 可以标记为 invalid patch。

### Soft Penalties

用于 reward shaping 和报告：

- 小幅 SLOC 增长；
- max CC 增长；
- nesting 增长；
- maintainability index 下降；
- 重复逻辑增加。

这些不应直接判 invalid，而应进入 complexity penalty。

## 推荐 MVP 指标

第一版不需要一次实现所有传统指标。推荐先实现：

```text
sloc
file_count
function_count
class_count
cyclomatic_complexity_max
cyclomatic_complexity_mean
max_function_length
max_nesting_depth
import_count
patch_added_lines
patch_deleted_lines
```

后续再加入：

```text
maintainability_index
halstead_volume
halstead_effort
duplication_score
global_mutable_state_count
large_literal_table_size
```

## 报告字段

每个 run 的 `metrics.json` 应包含：

```json
{
  "final_complexity": {
    "sloc": 386,
    "file_count": 1,
    "function_count": 18,
    "cyclomatic_complexity_max": 21,
    "max_nesting_depth": 6
  },
  "complexity_growth": {
    "sloc_delta_from_h0": 379,
    "cyclomatic_complexity_max_delta_from_h0": 18
  },
  "compression_epochs": 1,
  "complexity_penalty_total": 0.014
}
```

每个 epoch 的 `transition.json` 应包含：

```json
{
  "complexity_before": {},
  "complexity_after": {},
  "complexity_delta": {},
  "complexity_penalty": 0.001
}
```

## 解释方式

复杂度指标不应该替代 heldout score。主结论仍然应该基于：

```text
heldout learning AUC
final heldout mean return
best heldout mean return
```

复杂度用于回答另一个问题：

```text
这个 agent 是在稳健地改进 heuristic system，还是在用越来越脆弱的规则堆分数？
```

因此论文或报告中应同时展示：

- score curve；
- complexity curve；
- score vs complexity tradeoff；
- final vs best score；
- compression epochs。
