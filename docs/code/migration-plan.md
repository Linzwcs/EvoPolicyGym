# 迁移计划

## Phase 0: 建立安全网

- 添加 `pyproject.toml` 或最小测试配置。
- 保留现有 smoke 命令：

```text
PYTHONPATH=src python -m compileall src/hlbench
PYTHONPATH=src python -m hlbench.rollout.run_policy --scenario minigrid_doorkey --split train --episodes 2 --run-id smoke
PYTHONPATH=src python -m hlbench.harness.run_codex_step --scenario minigrid_doorkey --run-id codex_skip --skip-codex
```

`run_codex_step` 是当前兼容入口，不代表目标架构绑定 Codex。

- 为 scenario load、summary、policy compile、event logger 增加小单元测试。

## Phase 1: 抽出 core 数据模型

- 新增 `core.scenario`，从 `rollout.run_policy.Scenario` 迁移。
- 扩展 scenario schema，加入 `env_backend`、`env_kwargs`、`observation_mode`、`task`、`action_meanings`，旧 scenario 默认迁移为 `env_backend="gymnasium"`。
- 新增 `core.task`，合成 `EnvContract + ScenarioSpec + WorkspaceContract`，写出 `task_contract.json` 和 `task.md`。
- 新增 `core.events`，替代 `run_codex_step.EventLogger`。
- 新增 artifact 写入 helper，先保持输出格式不变。

验收：现有 CLI 输出 artifact 路径和 summary 字段不变。

## Phase 2: 抽象环境 backend

- 新增 `envs.base.EnvironmentBackend` 和 `EnvironmentInstance` 协议。
- 给 `EnvironmentBackend` 增加 `describe(spec) -> EnvContract`。
- 新增 `envs.gymnasium_backend`，覆盖 Gymnasium 标准环境。
- 新增 `envs.space_schema`，将 Gymnasium `observation_space` / `action_space` 转成 task contract schema。
- 将现有 MiniGrid 专用 adapter 改成 Gymnasium backend 的 observation wrapper。
- 增加至少一个非 MiniGrid Gymnasium smoke scenario，例如 CartPole，验证架构不绑定 MiniGrid。

验收：MiniGrid scenario 继续通过；新增 Gymnasium 标准环境可以不改 harness 主流程运行；生成的 `task_contract.json` 和 `task.md` 包含 observation 和 action schema。

## Phase 3: 重组 rollout

- 将 `run_episode`、`run_policy` 拆到 `rollout.engine`。
- `rollout.run_policy` 退化为兼容 CLI wrapper。
- 明确 `RolloutResult`，减少 runner 对文件布局的隐式依赖。

验收：DoorKey 和 KeyCorridor smoke rollout 仍通过。

## Phase 4: 对齐 workspace 契约

- 新 workspace 改为 `system/policy.py`、`feedback/`、`tools/`、`experiments/`。
- 添加 workspace contract 检查：可写路径、只读路径、protected file hash。
- 保留旧 workspace runner 兼容层，直到 harness 全部迁移完成。

验收：agent 只能编辑允许目录，train feedback 不泄露私有评估。

## Phase 5: 拆分 harness runner

- 抽出 `harness/agents/config.py`、`harness/agents/command.py`、`evaluator.py`、`reward.py`。
- 新增 agent backend 配置：`backend`、`command`、`timeout`、`env`、`final_json_required`。
- 将现有 `codex exec` 改为 `command` backend 的一个 preset。
- 为 Claude Code 预留 preset，但不在核心流程中依赖它。
- `epoch_runner.py` 负责单轮 transition。
- `loop_runner.py` 只负责多轮状态推进和 checkpoint。

验收：`run_codex_step` 和 `run_codex_loop` 不再互相导入私有函数；新增通用入口可以用不同 `--agent-command` 运行同一 benchmark。

## Phase 6: CLI 与报告

- 建立统一 CLI wrapper，保留旧模块路径兼容一段时间。
- 把 report / metrics 作为独立模块接入，不放进 epoch runner。

## 推进方式

每个 phase 单独提交。每次提交至少运行 `compileall` 和一个 `--skip-agent` harness smoke。任何 artifact schema 变更都必须同步更新 `docs/benchmark/evaluation_protocol.md`。
