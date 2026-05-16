# 可见性边界

HLBench 的可见性边界决定 benchmark 是否可信。

原则很简单：

```text
人类玩家或 policy author 在正常训练中能看到的东西，可以给 agent。
私有评测轨迹、隐藏答案、heldout 分数/失败细节和 evaluator internals，不能给 agent。validation 可以把 aggregate score 放进 workspace history，但不能暴露 seed、replay、逐 episode 记录或失败细节。
```

## Agent 可以看到

agent 可以看到 learner workspace 中的公开材料：

- `task.md`
- 当前 `system/policy.py`
- 允许编辑文件列表；
- 允许运行的 train rollout 命令；
- train split 的 rollout summary；
- train split 的失败摘要；
- train split 的 replay 或 trace 文件；
- 历史公开 train rollout；
- 历史 aggregate validation score；
- 自己在 `tools/` 和 `experiments/` 中写入的分析脚本、notes 和临时实验结果；
- 之前 epoch 的公开训练结果和 patch 历史。

如果游戏本身在交互时暴露 observation、action、reward、terminated、truncated、info 中的公开字段，这些也可以通过 train rollout 日志或 agent 自己写的探针观察。

## Agent 不能看到

agent 不能看到：

- validation seed list；
- heldout seed list；
- validation / heldout failure details；
- validation / heldout replay 或 trace；
- validation / heldout per-episode logs；
- heldout aggregate score；
- hidden simulator state；
- evaluator source code；
- reward implementation internals；
- solution policy；
- test answers；
- 私有评测输出目录；
- 文件系统里的隐藏评测结果；
- 任何可以反推出 heldout cases 的私有 artifact。

这里的 “不能看到” 不应该实现成“先记录下来但藏起来”。第一版 HLBench 中，validation / heldout 不应生成可回放材料、逐步轨迹或失败明细。benchmark 可以持久化聚合分数；validation aggregate 可以进入 agent-visible history，heldout aggregate 只用于最终报告和外部评估。

## 游戏分数与私有评测的区别

游戏环境自己给 reward。HLBench 不需要隐藏 reward 函数本身的结果：每个 train episode 的 reward 是正常反馈，agent 可以看到。

需要避免生成和暴露的是：

```text
这个 policy 在私有 validation / heldout seeds 上为什么失败
```

例如：

- train 上看见某个 seed 卡住，可以给失败 replay；
- validation / heldout 上某个 seed 卡住，不应生成可持久化 replay 或失败明细；
- train reward 可以被 agent 用来调试；
- validation 聚合分数可进入 history；heldout 聚合分数只用于报告和外部评估。

## Observation 暴露原则

对 Gymnasium 环境，agent 可以通过 policy 和 train rollout 看到环境正常返回的 observation。

不要额外提供：

- simulator 内部对象；
- grid 的私有全局状态，除非这本来就是 observation；
- wrapper 内部用于评分的状态；
- private seed 的逐步轨迹；
- hidden answer 或 oracle action。

如果环境的标准 observation 已经包含完整 grid 或局部视野，那它就是可见状态。benchmark 不应该为了人为增加难度而隐藏正常 observation，也不应该为了帮 agent 而暴露 observation 之外的内部状态。

## Replay 与私有分数策略

train replay 可以公开。validation / heldout 不生成 replay。

推荐：

```text
train:
  summary.json       public
  trials.jsonl       public
  failures.jsonl     public
  replay files       public

validation:
  score summary      workspace history / report
  per-episode logs   not generated
  failure details    not generated
  replay files       not generated

heldout:
  score summary      report only
  per-episode logs   not generated
  failure details    not generated
  replay files       not generated
```

如果未来需要给 agent 少量 validation feedback，应该作为单独 benchmark variant，并在报告中明确标注。默认 HLBench 不这样做。

## 文件系统隔离

runner 应该做到：

- agent 工作目录只包含公开 workspace；
- evaluator 目录不在 agent 可访问路径下，或至少不被 prompt 暴露；
- allowed tools 只能运行 train split；
- protected files 有 hash 校验；
- agent 不能修改 `task.md`、`feedback/`、scenario config、seed split 或 evaluator。

文件系统隔离是 benchmark 的实现责任，不能只依赖 prompt 约束。
