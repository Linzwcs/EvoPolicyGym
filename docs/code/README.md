# HLBench 代码重构设计

本目录用于梳理 `src/hlbench` 的全新代码架构与需求。当前先沉淀设计，不直接重写 `src/`，避免在协议未稳定前扩大改动面。

## 阅读顺序

1. [requirements.md](requirements.md): 重构后的系统需求、边界和验收标准。
2. [glossary.md](glossary.md): 核心术语定义。
3. [task-contract.md](task-contract.md): learner workspace 中 `task.md` 必须公开的环境、观测、动作和命令契约。
4. [visibility-and-data-boundary.md](visibility-and-data-boundary.md): agent-facing、benchmark-private、human-facing 数据边界。
5. [scenario-validation.md](scenario-validation.md): smoke / official scenario 准入规则。
6. [failure-and-scoring.md](failure-and-scoring.md): 执行错误、策略失败、协议违规和最低分规则。
7. [experiments-and-visualization.md](experiments-and-visualization.md): 面向人的实验分析、指标表和报告可视化边界。
8. [schema-design.md](schema-design.md): `ScenarioSpec`、`EnvContract`、`TaskContract`、`WorkspaceContract`、`AgentConfig` 和 `Transition` 的字段规范。
9. [open-questions.md](open-questions.md): 仍需决策的问题、当前倾向和优先级。
10. [architecture.md](architecture.md): 目标代码分层、模块职责和核心接口。
11. [migration-plan.md](migration-plan.md): 从当前原型迁移到目标架构的阶段计划。

## 当前判断

现有代码已经能跑 MiniGrid pilot，但还处在原型阶段：

- `rollout`、workspace 创建、agent 执行、评估、reward、artifact 写入混在少数大文件中。
- 现有入口以 `run_codex_*` 命名，容易把 benchmark 绑定到单一 coding agent；目标架构必须支持 Codex CLI、Claude Code 或任意命令式 agent backend。
- `run_codex_loop.py` 直接复用 `run_codex_step.py` 的私有函数，模块边界不稳定。
- 文档中的 workspace 契约是 `system/feedback/tools/experiments`，当前真实 workspace 仍是 `policy.py`、`rollout/`、`rollouts/` 的早期形态。
- scenario、environment backend、policy、split、artifact、event、reward 缺少统一数据模型，导致后续扩展环境和报告时容易复制逻辑。

## 设计原则

- 先稳定 benchmark 协议，再优化内部实现。
- 每个模块只拥有一种职责：环境适配、rollout、workspace、agent 执行、评估、artifact、报告分开。
- 环境 backend 是可插拔层；Gymnasium 是第一版通用后端，MiniGrid 是 Gymnasium 上的 scenario/wrapper，不是核心假设。
- official scenario 必须通过准入验证；smoke scenario 只能用于开发和连通性测试。
- task contract 必须明确 observation schema 和 action schema；不能让 agent 凭空猜动作含义或输入格式。
- agent backend 是可选插件，不是 benchmark 核心；核心流程只依赖统一的 agent 执行接口。
- 私有评估数据不能进入 learner workspace。
- failure taxonomy 和 minimum score 必须固定；崩溃、非法 action、协议违规都要可比较、可审计。
- 可视化和报告是 human-facing analysis，不进入 learner workspace；validation / heldout 不能泄露 per-episode、replay、seed 或 failure trace。
- CLI 只做参数解析和调用应用服务，不承载核心逻辑。
- 重构必须保持现有 smoke 命令可迁移、可验证。
