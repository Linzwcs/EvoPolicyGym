# Task Contract

`task.md` 是 learner workspace 中最重要的公开说明。它不应由 environment backend 直接写文件；正确边界是 environment backend 产出 `EnvContract`，scenario 补充任务语义，workspace builder 合成 `TaskContract`，再渲染 `task.md` 和 `task_contract.json`。目标是让 agent 明确知道能看到什么、能做什么、如何验证，而不是靠猜。

## 生成边界

```text
EnvironmentBackend.describe()
  -> EnvContract
       observation_schema
       action_schema
       reward_range
       termination semantics
       public info schema

ScenarioSpec
  -> task goal
     success condition
     reward description
     action meanings override
     scenario notes

Workspace/TaskContractBuilder
  -> task_contract.json
  -> task.md
```

Environment backend 只负责描述公开环境接口，不负责 benchmark 目标、允许命令、workspace 文件边界或私有评估规则。

## 必填内容

每个 `TaskContract` 至少包含：

- 环境定义：`scenario_id`、`env_backend`、`env_id`、最大步数、episode 终止条件。
- 任务目标：自然语言目标、success 条件、主要 score。
- 可见 observation：字段名、类型、shape、取值范围、含义和示例。
- 可执行 actions：action space 类型、合法取值、每个 action 的语义。
- reward 定义：reward 来源、稀疏/稠密程度、失败或 timeout 如何计分。
- policy 接口：`reset`、`act` 的签名、入参格式、返回值格式。
- 可运行命令：允许的 train rollout 命令、语法检查命令、输出位置。
- 数据边界：说明 validation 只暴露 aggregate history，heldout 不可见，不能尝试读取私有 evaluator artifact。

## Observation Schema

observation 必须描述 policy 实际可见的信息面。原则是：policy 看到的内容应与人类分析任务时看到的内容一致，不能给 policy 暴露人类不可见的 privileged simulator state。schema 应尽量机器可读：

HLBench 第一版支持 telemetry observation surface：

- `telemetry`: 人和 policy 都看同一组公开传感器读数，例如 Classic Control 的 CartPole、MountainCar、Pendulum、Acrobot。

Gymnasium Classic Control 官方任务默认是 telemetry。如果要研究 pixel-control，应拆成独立 scenario，例如 `mountain_car_pixels`，并新增专门的 image observation / frame artifact 支持，而不是在现有 telemetry scenario 中混用 render output。

```json
{
  "observation": {
    "type": "dict",
    "fields": {
      "position": {"type": "array", "shape": [2], "dtype": "float32"},
      "velocity": {"type": "array", "shape": [2], "dtype": "float32"}
    }
  }
}
```

对于 MiniGrid 一类任务，可以写：

```json
{
  "observation": {
    "type": "dict",
    "fields": {
      "image": {"type": "array", "shape": [7, 7, 3], "meaning": "egocentric visible grid"},
      "direction": {"type": "int", "meaning": "agent facing direction"},
      "mission": {"type": "string", "meaning": "natural language task instruction"}
    }
  }
}
```

未来视觉任务可以接收图片输入。例如自定义 pixel-control scenario 可以使用 image observation：

```json
{
  "observation": {
    "type": "dict",
    "fields": {
      "image": {
        "type": "array",
        "shape": [106, 160, 3],
        "dtype": "uint8",
        "color_space": "RGB",
        "source": "environment observation"
      }
    }
  }
}
```

未来实现 image observation 时，train replay 不应把每一帧内联到 JSONL。应把 frame 写成文件，并在 replay 中记录相对路径、shape 和 dtype。validation / heldout 仍然不生成 replay 或 frame 文件。

## Action Schema

action schema 必须说明合法 action 和语义。对于 discrete action，不能只写 `[0, n)`：

```json
{
  "action_space": {
    "type": "discrete",
    "n": 7,
    "actions": [
      {"id": 0, "name": "left", "meaning": "turn left"},
      {"id": 1, "name": "right", "meaning": "turn right"},
      {"id": 2, "name": "forward", "meaning": "move forward"},
      {"id": 3, "name": "pickup", "meaning": "pick up object in front"},
      {"id": 4, "name": "drop", "meaning": "drop carried object"},
      {"id": 5, "name": "toggle", "meaning": "open door or interact"},
      {"id": 6, "name": "done", "meaning": "signal done"}
    ]
  }
}
```

对于 continuous action，必须说明 shape、dtype、每个维度的范围和含义：

```json
{
  "action_space": {
    "type": "box",
    "shape": [2],
    "dtype": "float32",
    "low": [-1.0, -1.0],
    "high": [1.0, 1.0],
    "dimensions": [
      {"index": 0, "name": "steering"},
      {"index": 1, "name": "throttle"}
    ]
  }
}
```

如果 Gymnasium 环境无法自动提供 action meanings，scenario author 必须在 `scenario.json` 或 `task_spec.md` 中补充。缺少动作语义的任务不应进入正式 benchmark，只能作为低信号 smoke test。

## 输出文件

workspace 中应同时写出：

```text
task.md
task_contract.json
```

`task_contract.json` 是机器可读规范来源，`task.md` 是给 agent 阅读的 Markdown 渲染。两者语义必须一致。

## 生成来源

`TaskContract` 的内容来自三层：

- scenario metadata：任务目标、success、reward、动作语义补充。
- `EnvContract`：observation/action space 的类型、shape、范围、reward range、termination semantics。
- `WorkspaceContract`：policy 接口、允许命令、数据边界和文件布局。

生成后的 `task.md` 和 `task_contract.json` 都是只读文件。agent 修改它们应计为 workspace contract violation。
