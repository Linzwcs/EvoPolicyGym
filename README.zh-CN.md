# EvoPolicyGym

[English](README.md) | [中文](README.zh-CN.md)

EvoPolicyGym 是一套基准测试基础设施，用来评估 coding agent 能否在预算受限的环境反馈中迭代改进可执行策略。它采用类似 online judge 的协议：agent 在 workspace 中编辑代码，通过本地 API 提交 policy rollout，读取反馈工件，然后继续修改，直到 episode 预算耗尽。

## 评测目标

EvoPolicyGym 评测的是 agent 是否能把环境反馈转化为更好的可执行 policy 代码。基准不限定 policy 内部方法：提交的策略可以使用规则、搜索、规划、学习组件，或其它 Python 逻辑。真正被协议约束的是交互方式：真实环境 rollout 必须经过 EvoPolicyGym server，消耗 run 预算，并留下可复现的产物。

## 总体实验协议

每个 benchmark run 是一个闭环优化过程，包含三个角色：

- **Agent**：编辑 `workspace/system/`，决定何时 submit，并读取反馈工件。
- **Server**：快照 policy 代码，在 sandbox 中执行受控 rollout，写入反馈，维护预算，并完成最终评分。
- **Workspace**：在 `system/` 下暴露可写 policy 代码，在 `feedback/submit_NNN/` 下暴露只读反馈。

一次 run 的循环如下：

1. Agent 通过 `/info` 查询运行时状态，通过 `/task` 获取任务契约。
2. Agent 编辑 `system/`。
3. Agent 调用 `/submit`，指定一个或多个训练 `env_instances`。
4. Server 快照 policy，运行这些 episodes，扣除 episode budget，并在 `feedback/submit_NNN/` 下写入 `summary.json`、逐步 trajectory、可选视频、观测文件、stdout/stderr 和错误文件。
5. Agent 只能分析这些可见 train feedback，然后继续迭代，直到 episode budget 耗尽。
6. 当 `remaining_budget == 0` 时，server 自动 finalize：对所有 `status == "ok"` 的 checkpoint 在隐藏 validation cases 上评估，按 validation score 选择最佳 checkpoint，再把该 policy 放到隐藏 held-out cases 上评估，得到最终分数。

可见性边界是协议的核心：

- **优化期间可见**：任务文本、运行时预算状态、train `env_instance` ID，以及 `/submit` 产生的 train feedback。
- **优化期间隐藏**：validation cases、held-out cases、它们的 seeds、random/expert scoring anchors、validation scores、held-out returns，以及 final score。

所有用于优化的 rollout 数据都必须由 `/submit` 产生。Agent 可以做本地语法检查或静态分析，但不能在 server 控制的 submit 路径之外，通过本地 Gymnasium、MuJoCo、Box2D、HighwayEnv 或其它 simulator 额外生成环境 episodes。

主实验的 Core-16 配置位于 `config/main-128-*.toml`，每个 run 使用 128 个可见训练 episodes。协议默认的隐藏选择/评测池为：每个成功 checkpoint 用 64 个 validation episodes 做选择，对选出的 checkpoint 用 256 个 held-out episodes 做最终评分。规范协议见 [`docs/protocol/`](docs/protocol/)，Core-16 环境套件见 [`docs/envs/core_suite.md`](docs/envs/core_suite.md)。

## 状态

EvoPolicyGym 目前是 alpha 软件。活跃 package 位于 `src/evopolicygym`；冻结的 v1 材料位于 `archive/v1/`，仅供参考。

## 安装

```bash
uv sync
```

如果要运行主 Core-16 实验栈，请安装 Gymnasium 和兼容环境 family：

```bash
uv sync --extra dev --extra env-gym --extra env-compatible
```

同样的环境准备也可以通过脚本完成：

```bash
scripts/setup-env.sh --core
```

`--core` 会安装 `config/main-128-*.toml` 所需依赖：Gymnasium classic control、Box2D、MuJoCo、MiniGrid、HighwayEnv 和 Gymnasium-Robotics。小型 smoke 配置只需要基础 package：

```bash
scripts/setup-env.sh --smoke
```

可选环境 family 按 extra 拆分，只在需要时安装：

```bash
uv sync --extra env-visual
uv sync --extra env-multi
uv sync --extra env-web
uv sync --extra env-heavy
uv sync --extra env-jax
uv sync --extra env-mario
```

`env-jax` 和 `env-mario` 是分开的运行目标。Gymnasium 的 JAX 环境需要 `numpy>=2.1`，而 MO-Gymnasium 的 Mario extra 当前 pin 住 `numpy<2.0`，因此应在不同虚拟环境中测试。

### Runtime Assets

部分可选环境 family 需要非 Python 资产：

- BrowserGym MiniWoB++：运行 `scripts/setup-env.sh --core --web`。该命令会安装 `env-web`，把 `Farama-Foundation/miniwob-plusplus` checkout 到被忽略的 `third_party/miniwob-plusplus`，commit 为 `7fd85d71a4b60325c6585396ec4f48377d049838`，并安装 Playwright Chromium。EvoPolicyGym 会从仓库根目录自动检测该路径；如果在其它位置运行，请把 `MINIWOB_URL` 设为脚本打印的 `file://.../miniwob/` URL。
- Atari/ALE：使用 `scripts/setup-env.sh --core --atari-roms` 安装 Gymnasium assets；该命令会在当前 `.venv` 中运行 AutoROM。
- MiniGrid WFC assets 已 vendored 到 `src/evopolicygym/envs/gym/assets/minigrid_wfc_patterns/`，无需额外手动步骤。

更完整的可选环境路线图见 [`docs/envs/overview.md`](docs/envs/overview.md)。

## 快速开始

运行单元测试：

```bash
uv run python -m unittest discover -s tests
```

查看 CLI 帮助：

```bash
uv run evopolicygym --help
```

旧的 `feedbackgym` package 和 CLI 名称不再支持。

创建可复现的外部 case splits：

```bash
uv run evopolicygym data make \
  --env gym/taxi \
  --root data/gym/taxi \
  --seed 0 \
  --train-size 64 \
  --valid-size 64 \
  --heldout-size 256
```

运行一个本地 command-agent benchmark session：

```bash
uv run evopolicygym run \
  --env toy \
  --runs runs \
  --model script \
  --exp-id smoke-001 \
  --budget 8 \
  --agent command -- python agent.py
```

从 TOML/JSON config 运行：

```bash
uv run evopolicygym run --config docs/examples/cartpole-codex.toml
```

运行小型 live-agent smoke suite：

```bash
uv run evopolicygym suite --config config/smoke-8-suite.toml
```

运行 128-budget 主实验 suites：

```bash
uv run evopolicygym suite --config config/main-128-codex-suite.toml
uv run evopolicygym suite --config config/main-128-claude-suite.toml
uv run evopolicygym suite --config config/main-128-kimi-suite.toml
```

这些配置要求对应 CLI（`codex`、`claude` 或 `kimi`）已在本地认证。Codex 配置使用 `bypass = true`，使 harness 能访问本地 EvoPolicyGym HTTP API；server-side policy rollout 仍由 EvoPolicyGym 控制。

## 仓库结构

- `src/evopolicygym/`：package 源码。
- `config/`：已纳入版本控制的 smoke 和主实验 suite configs。
- `scripts/`：本地 setup helpers。
- `docs/protocol/`：规范性 protocol draft。
- `docs/envs/`：环境覆盖说明、路线图和 discovery 输出。
- `docs/examples/`：小型示例 configs 和 fixtures。
- `tests/`：基于标准库 `unittest` 的测试套件。
- `third_party/`：被忽略的本地 runtime assets，例如 MiniWoB++ HTML。
- `archive/v1/`：冻结的 legacy code、docs、analysis 和 v0 run data。

生成的 run outputs 应放在 `runs/` 或 `experiment/` 下，并默认被忽略。本地生成的 case splits 应放在被忽略的 `data/` 下；纳入版本控制的 fixtures 位于 `docs/examples/data/`。

## Agent Adapters

EvoPolicyGym 当前包含以下 adapters：

- 通用 persistent JSONL command agent；
- OpenAI Codex CLI；
- Claude Code；
- Kimi Code。

每个 adapter 都会在 benchmark turns 之间保留一个逻辑 agent session；server 负责控制 rollout budget、feedback artifacts、隐藏 validation 和最终 scoring。

## 环境发现

检查结构化 EvoPolicyGym environment manifest，并对已注册环境运行 smoke checks：

```bash
uv run evopolicygym check-envs
uv run evopolicygym check-envs --env gym/taxi
uv run evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1
uv run evopolicygym check-envs --discover --min-level L0
```

重新生成已安装环境 registry 报告：

```bash
uv run evopolicygym discover-envs \
  --output docs/envs/discovered.json \
  --markdown docs/envs/env_list.md
```

Discovery report 反映的是当前安装的可选 packages；它不承诺每个 discovered task 都已经拥有校准过的 EvoPolicyGym scoring setup。

Core-16 readiness checks 使用：

```bash
uv run evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1 --timeout 60
```

## 安全

EvoPolicyGym 会执行 agent 写出的 Python policies。请把 benchmark runs 视为不可信代码执行。对 live agent experiments 使用 sandbox、隔离 workspace 和一次性 credentials。详见 [`SECURITY.md`](SECURITY.md)。

## Citation

```bibtex
@software{evopolicygym2026,
  title  = {EvoPolicyGym},
  author = {Zhilin Wang and Han Song and Runzhe Zhan and Jusen Du and Jiacheng Chen and Tianle Li and Qingyu Yin and Yulun Wu and Zhennan Shen and Tong Zhu and Yanshu Li and Guanjie Chen and Derek F. Wong and Yafu Li and Yu Cheng and Yang Yang},
  year   = {2026},
  url    = {https://github.com/Linzwcs/EvoPolicyGym}
}
```

## License

EvoPolicyGym 基于 MIT License 发布。详见 [`LICENSE`](LICENSE)。
