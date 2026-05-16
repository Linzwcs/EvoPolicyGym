# 环境支持 Roadmap

HLBench 的环境集合应保持小而覆盖面清晰。第一版目标不是覆盖 Gymnasium 全量环境，而是建立一个可复现、可解释、可横向比较的 16 环境核心套件：4 类能力，每类 4 个环境。

## 选择原则

- 每个环境必须能清楚描述 observation、action、reward、success 和 timeout。
- scenario 默认使用官方环境给出的 observation surface，不为了统一格式而强行改成状态参数或图片。
- 每个 scenario 必须标记 observation 类型：`state` 表示低维状态参数，`symbolic` 表示结构化 grid/mission，`image` 表示图片输入。
- agent 可见信息必须等同于该 scenario 声明的 public observation。
- train / validation / heldout 使用共享随机 seed pool，split 不重叠。
- validation 只允许暴露 aggregate summary；heldout 不进入 learner workspace。
- 每个 official scenario 必须有 baseline policy、随机 baseline、参考分数和 smoke command。
- 跨环境总榜使用 normalized score；单环境报告保留 raw return。

## 目标 16 环境

### 1. Classic Control / Telemetry

这类环境验证低维传感器、离散动作、连续动作和长 horizon 的基础协议。

| Scenario | Gymnasium env | Observation | Action | 主要能力 |
| --- | --- | --- | --- | --- |
| `cartpole_balance` | `CartPole-v1` | state | discrete | 最小 sanity check，正 reward，长 episode |
| `mountain_car` | `MountainCar-v0` | state | discrete | momentum 策略，稀疏成功信号 |
| `acrobot_swingup` | `Acrobot-v1` | state | discrete | 更长 horizon 的 swing-up 控制 |
| `pendulum_swingup` | `Pendulum-v1` | state | continuous | 连续动作、action clipping、负 cost |

### 2. Box2D Control

这类环境更接近常见 DeepRL benchmark，覆盖复杂动力学和更高维动作。

| Scenario | Gymnasium env | Observation | Action | 主要能力 |
| --- | --- | --- | --- | --- |
| `lunar_lander` | `LunarLander-v3` | state | discrete | 标准离散控制对照 |
| `lunar_lander_continuous` | `LunarLanderContinuous-v3` | state | continuous | 连续推力控制 |
| `bipedal_walker` | `BipedalWalker-v3` | state | continuous | 高维连续 locomotion |
| `car_racing` | `CarRacing-v3` | image | continuous | 官方 RGB 图像输入、轨迹控制 |

### 3. MiniGrid / Programmatic Reasoning

这类环境验证结构化 observation、自然语言 mission、记忆、探索和程序式规划。

| Scenario | Gymnasium env | Observation | Action | 主要能力 |
| --- | --- | --- | --- | --- |
| `minigrid_doorkey_16x16` | `MiniGrid-DoorKey-16x16-v0` | symbolic | discrete | 更大地图探索 |
| `minigrid_keycorridor_s6r3` | `MiniGrid-KeyCorridorS6R3-v0` | symbolic | discrete | 大走廊、多房间、任务目标检索 |
| `minigrid_obstructedmaze_2dlhb` | `MiniGrid-ObstructedMaze-2Dlhb-v1` | symbolic | discrete | 阻塞门、物体操作、多步解谜 |
| `minigrid_lavacrossing_s11n5` | `MiniGrid-LavaCrossingS11N5-v0` | symbolic | discrete | 大图安全导航和稀疏成功 |

### 4. MuJoCo Continuous Control

这类环境用于和标准 DeepRL 方法对齐，测试高维连续控制上的可扩展性。

| Scenario | Gymnasium env | Observation | Action | 主要能力 |
| --- | --- | --- | --- | --- |
| `reacher` | `Reacher-v5` | state | continuous | 短 horizon 精确控制 |
| `inverted_pendulum` | `InvertedPendulum-v5` | state | continuous | 稳定控制 sanity check |
| `hopper` | `Hopper-v5` | state | continuous | locomotion 入门 |
| `half_cheetah` | `HalfCheetah-v5` | state | continuous | 常见 DeepRL 横向对照 |

## 支持阶段

### Phase 0: Current Smoke Suite

目标：把现有轻量 scenario 全部跑通。

- 已有：`cartpole_balance`、`mountain_car`、`acrobot_swingup`、`pendulum_swingup`。
- 已接入 Box2D smoke：`lunar_lander`、`lunar_lander_continuous`、`bipedal_walker`、`car_racing`。
- 每个环境先跑 `2 epoch` agent smoke。
- `mountain_car` 保留为当前主流程回归测试。
- `pendulum_swingup` 必须验证 continuous action schema、range check 和 minimum score。

### Phase 1: Officialize Classic Control

目标：将 Classic Control 从 smoke 提升为第一批 official。

- 补齐 reference baselines：random、initial heuristic、best known HLBench run、常见 DeepRL 参考。
- 固定 train / validation / heldout seed pool。
- 明确 raw return、success threshold、minimum score 和 normalized score。
- 为每个 scenario 增加标准脚本，例如 `scripts/run_codex_<scenario>_32.sh`。

### Phase 2: Add Box2D

目标：接入 `LunarLander`、`BipedalWalker` 和 `CarRacing`，验证更复杂动力学和第一类 image observation。

- `lunar_lander` 和 `lunar_lander_continuous` 已接入。
- `bipedal_walker` 已接入。
- `car_racing` 已接入，使用官方 RGB image observation，用来验证 image schema、image artifact 和 agent 可见输入协议。
- 文档必须注明依赖：`gymnasium[box2d]`。

### Phase 3: Add MiniGrid

目标：形成“控制 + 推理导航”双主线。

- `minigrid_public` wrapper 已实现，公开 `image`、`direction`、`mission` 和 `action_count`。
- 已接入 hard MiniGrid core：`minigrid_doorkey_16x16`、`minigrid_keycorridor_s6r3`、`minigrid_obstructedmaze_2dlhb`、`minigrid_lavacrossing_s11n5`。
- task contract 必须公开 grid encoding、mission、方向、可执行动作和对象语义。
- train replay 可以面向人类可视化；validation / heldout 不生成 replay。
- 从 legacy 中迁移 DoorKey 和 KeyCorridor 经验，但不复用旧 harness。

### Phase 4: Add MuJoCo

目标：覆盖高维连续控制。

- MuJoCo 先接 `reacher`、`inverted_pendulum`，再接 locomotion。

### Phase 5: Extend To Atari

目标：在 Core 16 的 `car_racing` 图片协议稳定后增加 Atari 扩展套件，而不是混入第一版核心榜单。

- 先固定 image observation protocol：分辨率、灰度/RGB、frame skip、frame stack、action repeat 和 max episode steps。
- 先选少量 Atari 代表任务，例如 `Pong`、`Breakout`、`Freeway`、`Seaquest`，再扩展到更多游戏。
- Atari scenario 必须写清楚 action meanings；不能只暴露 action id。
- train replay 可以保存抽样视频或帧摘要；validation / heldout 仍只保留 aggregate summary。
- 单独报告 Atari category score，避免和 Core 16 直接混合解释。

## 暂不纳入 Core 16

- Atari：作为 Phase 5 扩展套件，等 image protocol、artifact 大小限制和运行成本控制稳定后再接入。
- ToyText：适合调试 tabular policy，但和 HLBench 的持续 heuristic 优化主线弱相关。
- Robotics / GoalEnv：目标接口和成功语义更复杂，放到 MuJoCo 稳定之后。
- 纯随机生成任务集合：适合压力测试，但不适合作为第一版横向排行榜。

## Benchmark 使用方式

正式报告应同时给出：

- per-environment heldout learning curve；
- per-category normalized score；
- 16-env normalized aggregate；
- failure / timeout / minimum-score 统计；
- policy complexity growth；
- agent wall time 和 train episode budget。

单环境分数只能说明局部能力。HLBench 的核心指标应是：模型能否在四类环境上都稳定改进，并且不通过过度复杂、不可维护或泄露私有数据的策略取得分数。
