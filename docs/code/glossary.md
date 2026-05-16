# Glossary

## Transition

一次外层学习步骤，表示当前 heuristic system 从 `H_t` 变为 `H_{t+1}`。一个 epoch 通常产生一个 transition。

## Minimum Score

本轮 transition 出现执行错误或协议违规后使用的最低分。以前草案中称为 `floor`，后续文档统一使用 `minimum score`。

## Invalid Transition

本轮 transition 无效。常见原因包括 policy 编译失败、非法 action、agent 执行失败、修改只读文件或访问私有数据。invalid transition 应用 minimum score。

## EnvContract

由 environment backend 生成的公开环境接口描述，包括 observation schema、action schema、reward range、termination semantics 和 public info schema。

## ScenarioSpec

scenario 的静态配置，包括任务目标、环境后端、seed split、success 条件、reward 说明和 observation/action 语义补充。

## WorkspaceContract

workspace 文件边界和命令约束，包括可编辑路径、只读路径、policy 路径、feedback 布局、允许命令和私有数据规则。

## TaskContract

面向 agent 的完整任务契约，由 `EnvContract + ScenarioSpec + WorkspaceContract` 合成。`task_contract.json` 是机器可读规范来源，`task.md` 是 Markdown 渲染。

## Public Train Feedback

agent 可以看到的训练反馈，包括 train summary、failure summary 和可选 replay。只能来自 train split。

## Private Evaluation

benchmark 在 workspace 外部执行的 validation / heldout 评估。agent 不能看到 seeds、replays、failure traces 或 per-episode private details。
