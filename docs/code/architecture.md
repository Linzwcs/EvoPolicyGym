# 目标代码架构

## 包结构草案

```text
src/hlbench/
  core/
    paths.py
    scenario.py
    task_contract.py
    policy.py
    events.py
    artifacts.py
  envs/
    base.py
    registry.py
    gymnasium_backend.py
    space_schema.py
    wrappers.py
  rollout/
    engine.py
    records.py
    summarize.py
    cli.py
  workspace/
    contract.py
    create.py
    feedback.py
  harness/
    agents/
      command.py
      config.py
    evaluator.py
    reward.py
    epoch_runner.py
    loop_runner.py
    cli.py
  reports/
    builder.py
  scenarios/
    cartpole_balance/
    mountain_car/
    pendulum_swingup/
    acrobot_swingup/
```

## 模块职责

- `core.scenario`: 加载和校验 scenario JSON，提供 split seed 查询。
- `core.task`: 生成 learner 可见的任务契约，包括 observation schema、action schema、reward 和允许命令。
- `core.policy`: 加载 policy、校验接口、封装 policy 调用错误。
- `core.events`: JSONL event logger，不关心具体 run 类型。
- `core.artifacts`: 统一写 JSON、JSONL、patch、snapshot，避免散落文件写入。
- `envs.base`: 定义环境 backend 协议，与具体环境库解耦。
- `envs.registry`: 根据 scenario 的 `env_backend` 选择 backend 实现。
- `envs.gymnasium_backend`: 通用 Gymnasium backend，支持所有遵循 Gymnasium API 的环境。
- `envs.space_schema`: 将 Gymnasium `spaces` 或其他 backend space 描述转换为统一 schema。
- `envs.wrappers`: policy 可见 observation/action/info 的公开面转换，例如 MiniGrid public observation、通用 JSON 化或 rendered RGB image。
- `rollout.engine`: 运行 episode / rollout，返回结构化 `RolloutResult`。
- `workspace.create`: 按 workspace 契约生成 learner workspace。
- `workspace.feedback`: 把公开 train feedback 投放到 workspace。
- `harness.agents.command`: 执行任意命令式 agent，例如 Codex CLI、Claude Code、本地脚本。
- `harness.agents.config`: 提供 backend / preset / command 配置，不把核心逻辑绑定到具体产品。
- `harness.evaluator`: 统一 train / validation / heldout 私有评估。
- `harness.epoch_runner`: 执行单轮 transition。
- `harness.loop_runner`: 管理多轮、checkpoint、learning curve。
- `harness.reward`: 计算 reward component 和 accepted/rejected 标签。
- `reports.builder`: 从 transition records 生成 `learning_curve.json`、`metrics.json` 和静态 HTML report。

## Agent Backend 抽象

benchmark 只关心 agent 是否在 workspace 中完成一次受控修改，不关心它来自哪个产品。所有 backend 都实现同一协议：

```python
class AgentBackend(Protocol):
    name: str

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        ...
```

`AgentRunRequest` 至少包含 workspace、prompt path、timeout、环境变量和 run artifact 目录。`AgentRunResult` 必须保留原始 stdout/stderr；final JSON 是可选增强，不是 agent backend 成功的唯一标准。

第一版 backend：

- `command`: 通用命令执行器，适配 `codex exec`、`claude ...`、shell wrapper 或本地脚本。
- `agent-preset=none`: 不运行外部 agent，用于 evaluator smoke test。
- `agent-preset=codex`: `command=("codex", "exec")` 的便捷配置。
- `agent-preset=claude`: Claude Code 命令模板的便捷配置。

每轮 submission 目录保存 `agent.json`、`stdout.txt` 和 `stderr.txt`。`agent.json`
只记录 metadata 和相对输出路径，不内联长 stdout/stderr。

核心 runner 不导入具体 agent SDK，也不包含产品专属逻辑。

## Environment Backend 抽象

环境和 agent 一样必须可插拔。rollout engine 只依赖统一协议：

```python
class EnvironmentBackend(Protocol):
    name: str

    def describe(self, spec: ScenarioSpec) -> "EnvContract":
        ...

    def make(self, spec: ScenarioSpec) -> "EnvironmentInstance":
        ...

class EnvironmentInstance(Protocol):
    @property
    def action_count(self) -> int | None:
        ...

    def reset(self, seed: int, config: dict[str, Any]) -> dict[str, Any]:
        ...

    def step(self, action: Any) -> StepResult:
        ...

    def close(self) -> None:
        ...
```

第一版实现 `gymnasium` backend：

- 使用 `gym.make(spec.env_id, **spec.env_kwargs)` 创建环境。
- 用 `reset(seed=seed)` 控制复现。
- 将 reward、terminated、truncated、info 转成公共记录，并把 policy observation 转成契约定义的公开面。
- 从 policy-visible observation 和 `action_space` 生成基础 schema。
- 通过 wrapper 配置控制 observation surface，例如 `jsonable`、`minigrid_public`。
- Classic Control 等官方 telemetry 任务默认使用环境定义的数值 observation；pixel-control 应作为独立 scenario，并新增专门的 image observation 支持。

非 Gym 环境只要实现同一协议即可接入，例如 HTTP simulator、文本交互任务或自定义游戏引擎。harness、agent backend、reward 和 artifact 层不应感知具体环境类型。

## Task Contract 生成

`EnvContract` 是 environment backend 的输出；`TaskContract` 是 benchmark 面向 agent 的完整任务契约。environment backend 不直接写 `task.md`。

```text
EnvironmentBackend.describe() -> EnvContract
ScenarioSpec                  -> task semantics
WorkspaceContract             -> files and commands
TaskContractBuilder           -> task_contract.json + task.md
```

`task.md` 是 workspace 的公共操作说明，由 `core.task` 渲染。它聚合三类来源：

- scenario metadata：任务目标、success 条件、reward 说明、动作语义补充。
- `EnvContract`：observation/action space 的类型、shape、dtype、范围、reward range、termination semantics。
- `WorkspaceContract`：policy 接口、允许命令、反馈文件布局和私有数据边界。

核心数据结构：

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str
    shape: list[int] | None
    dtype: str | None
    meaning: str

@dataclass(frozen=True)
class ActionSpec:
    type: str
    n: int | None
    shape: list[int] | None
    low: Any | None
    high: Any | None
    actions: list[dict[str, Any]]

@dataclass(frozen=True)
class EnvContract:
    backend: str
    env_id: str
    observation_fields: list[FieldSpec]
    action_spec: ActionSpec
    reward_range: tuple[float | None, float | None]
    termination: dict[str, Any]
    public_info_schema: dict[str, Any]

@dataclass(frozen=True)
class TaskContract:
    scenario: ScenarioSpec
    env: EnvContract
    goal: str
    success_condition: str
    reward_description: str
    policy_interface: str
    allowed_commands: list[str]
```

Gymnasium backend 可以自动给出 action space 的类型和范围，但动作语义不总是可推断。正式 benchmark scenario 必须补齐 action meanings；否则 agent 只能知道动作合法性，不能可靠理解动作效果。

## 核心数据对象

优先使用小型 dataclass，减少自由字典在模块间传播：

```python
@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    scenario_id: str
    env_backend: str
    env_id: str
    env_kwargs: dict[str, Any]
    observation_mode: str
    task: dict[str, Any]
    action_meanings: list[dict[str, Any]]
    max_steps: int
    splits: dict[str, SplitSpec]

@dataclass(frozen=True)
class SplitSpec:
    seed_pool: str
    seed_count: int
    public_feedback: bool
    env_overrides: dict[str, Any]

@dataclass(frozen=True)
class RolloutResult:
    run_dir: Path
    summary: dict[str, Any]
    failures: list[dict[str, Any]]
    trials_path: Path | None
    replay_dir: Path | None

@dataclass(frozen=True)
class AgentRunResult:
    backend: str
    command: list[str]
    returncode: int
    timed_out: bool
    final_json: dict[str, Any]
    artifacts: dict[str, Any]
```

## 执行流

```text
LoopRunner
  -> WorkspaceCreator creates H_0
  -> for epoch:
       FeedbackPublisher.publish_previous_train_or_empty()
       TaskContractBuilder.write_task_files()
       AgentBackend.run()
       SubmissionStore.snapshot_policy_and_agent_result()
       Evaluator.standard_evaluation()
       RewardCalculator.compute()
       ArtifactWriter.write_transition()
       CheckpointStore.save_submission()
```

`LoopRunner` 不直接调用 environment；environment 只通过 `rollout.engine` 被访问。CLI 不写业务逻辑，只实例化配置并调用 runner。

## 关键边界

- learner workspace 是公共界面，run directory 是 benchmark 私有界面。
- replay 只属于 train feedback。
- run directory 中 `evaluation/train/` 存 train artifacts；`evaluation/validation` 和 `evaluation/heldout` 只存 aggregate summaries。
- scenario seed split 由 benchmark 拥有，workspace 内不应出现 validation / heldout seed。
- environment backend 负责公开观测面转换和 `EnvContract`，但不负责 rollout artifact、benchmark 目标或 agent workspace 文件写入。
- task contract 是 agent 的主要环境说明；没有 observation/action schema 的任务不应进入正式评测。
- 任何 protected file violation 都由 workspace contract 检查器报告，不散落在 runner 中。
- agent backend 不拥有 evaluator、reward 或 checkpoint 逻辑；它只能在 learner workspace 内工作。
