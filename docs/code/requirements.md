# 代码重构需求

## 目标

重构后的 `src/hlbench` 应从 MiniGrid 原型演进为清晰、可测试、可扩展的 benchmark 框架。第一阶段应支持 Gymnasium 的通用环境接口，MiniGrid 只是首批 scenario，不应成为核心架构假设。

## 功能需求

### Scenario 管理

- 从 `src/hlbench/scenarios/<name>/scenario.json` 加载 scenario。
- 统一暴露 `scenario_id`、`env_backend`、`env_id`、`max_steps`、`train/validation/heldout` split metadata。
- seed pool 必须是 benchmark-level 共享生成文件，例如 `src/hlbench/seeds/default/train.json`、`validation.json`、`heldout.json`；`scenario.json` 只引用 pool 名称。
- seed 文件应由固定 `generator_seed` 随机生成并随机切分，避免使用可猜测的连续区间。
- `env_backend` 用于选择环境后端，例如 `gymnasium`；`env_id` 是该后端内部环境 ID，例如 `MiniGrid-DoorKey-16x16-v0` 或 `CartPole-v1`。
- scenario 可以声明 observation/action/reward 的公开说明，以及可选 wrapper 配置。
- 支持每个 scenario 附带 baseline `policy.py`、`task_spec.md` 和 README。

推荐 scenario 配置骨架：

```json
{
  "scenario_id": "cartpole_balance_v0",
  "env_backend": "gymnasium",
  "env_id": "CartPole-v1",
  "max_steps": 500,
  "observation_mode": "jsonable",
  "task": {
    "goal": "Keep the pole balanced for as long as possible.",
    "success_condition": "Episode reaches the max step limit."
  },
  "action_meanings": [
    {"id": 0, "name": "push_left", "meaning": "push cart left"},
    {"id": 1, "name": "push_right", "meaning": "push cart right"}
  ],
  "splits": {
    "train": {"seed_pool": "default/train", "public_feedback": true},
    "validation": {"seed_pool": "default/validation", "public_feedback": false},
    "heldout": {"seed_pool": "default/heldout", "public_feedback": false}
  }
}
```

### Environment Backend

- 环境必须通过统一 backend 协议接入，rollout engine 不能直接依赖 Gymnasium 或 MiniGrid。
- 第一版必须实现通用 `gymnasium` backend，能够适配 Gymnasium 标准 `reset(seed=...)` 和 `step(action)` 接口。
- MiniGrid 只通过 Gymnasium backend 的 observation wrapper 或 scenario 配置适配，不作为唯一环境类型。
- 后续应能增加非 Gymnasium backend，例如自定义 HTTP 环境、本地 simulator、文本任务或游戏引擎，而不改 harness 主流程。
- backend 必须返回契约定义的 policy-visible observation、reward、terminated、truncated、info；视觉 observation 可以是 array-like image，replay 中用 frame 文件引用避免内联巨大像素数组。
- backend 必须隐藏 benchmark 私有数据，例如 validation / heldout seed、环境内部对象坐标和 evaluator 状态。
- backend 必须提供 `EnvContract`，描述 policy 实际可见的 observation schema、action schema、reward range、termination semantics 和 public info schema。
- policy observation 必须与 scenario 声明的公开信息一致；Classic Control 这类官方 telemetry 任务使用数值传感器读数，视觉任务可以使用 RGB image observation，但应作为独立 scenario 明确声明。
- backend 不得直接写 `task.md`；文件写入由 workspace builder 负责。

### Task Contract

- 每个 learner workspace 必须包含只读 `task.md` 和 `task_contract.json`，明确说明环境定义、policy 可见数据格式和可执行动作。
- `TaskContract` 由 `EnvContract + ScenarioSpec + WorkspaceContract` 合成。
- `task_contract.json` 是机器可读规范来源，`task.md` 是给 agent 阅读的 Markdown 渲染。
- `TaskContract` 必须列出 observation schema：字段名、类型、shape、取值范围、含义和示例。
- `TaskContract` 必须列出 action schema：action space 类型、合法取值、动作名称和动作语义。
- 对 discrete action，不能只暴露 `0..n-1`；必须尽量提供每个 action id 的含义。
- 对 continuous action，必须说明 shape、dtype、low/high 范围和每个维度的语义。
- 如果 Gymnasium 环境不能自动提供动作语义，scenario author 必须在 scenario metadata 中补充。
- `TaskContract` 还必须说明 reward、success 条件、policy 接口、允许 train rollout 命令和数据可见性边界。
- 缺少 action 或 observation 说明的 scenario 不能进入正式 benchmark，只能作为开发 smoke test。

### Policy 执行

- policy 必须通过明确接口加载和执行。
- 当前兼容接口为：

```python
from typing import Any

class Policy:
    def reset(self, task_config: dict) -> None: ...
    def act(self, observation: Any, context: dict) -> Any: ...
```

- evaluator seed 不得传给 policy；policy 只能看到 observation 和公开 task context。
- action 返回值由 `ActionSpec` 校验；MiniGrid 等 discrete 环境通常返回 `int`，continuous 或复合 action space 可以返回 list/dict/float。
- policy 异常、非法 action、timeout 都必须被记录为可汇总失败。

### Rollout 与反馈

- rollout engine 接收 scenario、split、policy path、episode 数、sampler seed 和 replay 开关。
- `--episodes` 表示从固定 seed pool 中抽样多少个 episode，本身不定义 split。
- `generator_seed`、seed pool 名称、seed 文件路径和真实 seed 值都不得出现在 workspace 或 agent 可见 contract 中。
- rollout engine 通过 `EnvironmentBackend` 创建环境实例，不直接调用具体环境库。
- rollout engine 必须用 `ActionSpec` 校验 policy action，非法 action 需要记录 failure mode。
- train rollout 可以写 replay、trial、failure 明细。
- validation / heldout 只允许生成聚合摘要；不得生成 replay、trace、episodes、failure details、action sequence 或 observation sequence。

### Failure 与 Scoring

- compile/import/missing policy、runtime exception、invalid action 都直接触发本轮 minimum score。
- agent backend 命令失败、超时、非零退出码或超过输出限制都直接触发本轮 minimum score。
- final JSON 缺失或不可解析只有在 `final_json_required=true` 时触发 minimum score；否则只记录 warning。
- timeout/truncation 是环境自然结果，不算执行错误，使用环境实际 return，但 `success=false`。
- 修改只读文件、访问私有数据或破坏 workspace contract 必须标记 `invalid_transition=true` 并触发 minimum score。
- 严重协议违规可以进一步触发 run disqualification。
- `transition.json` 必须记录 policy status、agent status、contract violations、minimum score applied 和 reward components。

### Workspace 生命周期

- 新 workspace 应对齐文档契约：

```text
workspace/
  AGENTS.md
  task.md
  task_contract.json
  system/policy.py
  feedback/
  tools/
  experiments/
```

- `AGENTS.md`、`task.md`、`task_contract.json`、`feedback/` 为只读协议面。
- agent 只能修改 `system/`、`tools/`、`experiments/`。
- `task.md` 和 `task_contract.json` 由 workspace builder 根据 `EnvContract`、`ScenarioSpec` 和 `WorkspaceContract` 共同生成。

### Agent 与 Harness

- harness 不得绑定 Codex、Claude Code 或任何单一 agent 产品。
- agent backend 必须是可配置项。第一版至少支持通用命令 backend；Codex CLI、Claude Code 等只作为 backend preset 或 adapter。
- agent runner 只负责执行外部命令、超时、stdout/stderr 和 artifact 记录；final JSON 解析是可选扩展。
- backend 需要统一输出 `AgentRunResult`，包含 provider/backend 名称、命令、退出码、是否超时、stdout/stderr 路径和可选 final JSON。
- stdout/stderr 必须写入 submission 目录中的文本文件，`agent.json` 只保存 metadata 和相对路径。
- epoch runner 负责一次 `H_t -> H_{t+1}`。
- loop runner 负责多 epoch、checkpoint、learning curve 和 transitions。

推荐配置形态：

```text
--agent-backend command --agent-preset none
--agent-backend command --agent-preset codex
--agent-backend command --agent-preset claude
--agent-backend command --agent-command "python tools/local_agent.py"
```

benchmark 不解释 agent 的内部能力，只检查 workspace 修改、输出 artifact 和协议违规。

### Artifacts 与报告

- 所有 run 级 artifact 写在 `runs/<model_name>/<env_name>/<run_id>/`。
- 每轮必须写 `transition.json`、`epoch.json`、事件日志和 evaluation 摘要。
- loop 结束后生成 `transitions.jsonl`、`learning_curve.json`、`metrics.json` 和 `report/index.html`。
- 报告由 `reports.builder` 独立生成，不能耦合进 epoch runner，也不能复制进 learner workspace。

## 非功能需求

- 可复现：相同 scenario、policy、seed split 应产生一致 summary。
- 可测试：核心逻辑不依赖 CLI，可用单元测试直接调用。
- 环境无关：Gymnasium 全部标准环境应通过同一 backend 接入；新增非 Gym 环境时只新增 backend/adapter 和 scenario，不改 runner 主流程。
- 任务自描述：agent 必须能从 `task.md` 理解输入格式、动作空间、目标和允许命令。
- agent 无关：更换 Codex CLI、Claude Code 或本地脚本不应影响 rollout、evaluation、reward、artifact schema。
- 错误可比较：执行错误、策略失败和协议违规都有固定分类和最低分规则。
- 可审计：每次 agent 执行、policy diff、reward 组件都有 artifact。

## 暂不做

- 不做大规模训练或 RFT。
- 不把 validation / heldout 细节暴露给 agent。
- 不做 suite-level dashboard 或跨 run 排名；当前只生成单 run report。
