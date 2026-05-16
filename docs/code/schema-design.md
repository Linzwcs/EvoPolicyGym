# Schema Design

本文件定义重构时优先稳定的核心 schema。实现时可以先用 dataclass + validator，之后再落成 JSON Schema 文件。

## Schema 清单

- `scenario.schema.json`: scenario 静态配置。
- `env_contract.schema.json`: environment backend 公开接口描述。
- `task_contract.schema.json`: agent 可见任务契约。
- `workspace_contract.schema.json`: workspace 文件边界和允许命令。
- `agent_config.schema.json`: agent backend 执行配置。
- `transition.schema.json`: 单轮 `H_t -> H_{t+1}` 结果记录。
- `experiment_manifest.schema.json`: agent 或 benchmark 实验记录。
- `report_metrics.schema.json`: final report 使用的汇总指标。

## ScenarioSpec

```json
{
  "name": "cartpole_balance",
  "scenario_id": "cartpole_balance_v0",
  "scenario_level": "official",
  "env_backend": "gymnasium",
  "env_id": "CartPole-v1",
  "env_kwargs": {},
  "observation_mode": "jsonable",
  "observation_options": {},
  "task": {
    "goal": "Keep the pole balanced.",
    "success_condition": "Episode reaches max_steps.",
    "reward_description": "Environment reward per step while balanced."
  },
  "observation_meanings": [],
  "action_meanings": [
    {"id": 0, "name": "push_left", "meaning": "push cart left"},
    {"id": 1, "name": "push_right", "meaning": "push cart right"}
  ],
  "max_steps": 500,
  "splits": {
    "train": {
      "seed_pool": "default/train",
      "public_feedback": true
    },
    "validation": {
      "seed_pool": "default/validation",
      "public_feedback": false
    },
    "heldout": {
      "seed_pool": "default/heldout",
      "public_feedback": false
    }
  },
  "minimum_score": {
    "performance_score": 0.0,
    "total_reward": -1.0
  }
}
```

Seed pools are benchmark-level shared files, not per-scenario files and not
inline in `scenario.json`:

```json
{
  "version": 1,
  "pool": "default",
  "split": "train",
  "generator": {
    "method": "random_partition",
    "generator_seed": 20260515,
    "min_seed": 0,
    "max_seed": 2147483647,
    "split_counts": {"train": 10000, "validation": 200, "heldout": 200}
  },
  "seeds": [0, 1, 2]
}
```

`--episodes` is a sampling budget over the fixed shared seed pool, not the split
definition. Train pools should be large enough for diverse feedback; validation
and heldout pools remain private and are sampled by the benchmark runner.
The generator seed, seed pool name, and seed files are benchmark-private and
must not appear in workspace task contracts.

Official scenarios must provide enough `observation_meanings` and `action_meanings` for an agent to understand the public interface. Automatically inferred space shape is not enough for official benchmark tasks. `observation_mode` defines what the policy actually sees; Classic Control tasks default to official numeric telemetry, while pixel-control tasks should be separate scenarios with dedicated image observation support.

## EnvContract

Produced by `EnvironmentBackend.describe(spec)`.

`observation_schema` describes the policy-visible observation. For standard Gymnasium telemetry tasks this is usually the raw observation space; for visual variants it should describe an image input generated from `env.render()`.

```json
{
  "backend": "gymnasium",
  "env_id": "CartPole-v1",
  "observation_schema": {
    "type": "array",
    "shape": [4],
    "dtype": "float32",
    "low": [-4.8, null, -0.418, null],
    "high": [4.8, null, 0.418, null],
    "dimensions": [
      {"index": 0, "name": "cart_position"},
      {"index": 1, "name": "cart_velocity"},
      {"index": 2, "name": "pole_angle"},
      {"index": 3, "name": "pole_angular_velocity"}
    ]
  },
  "action_schema": {
    "type": "discrete",
    "n": 2,
    "actions": [
      {"id": 0, "name": "push_left"},
      {"id": 1, "name": "push_right"}
    ]
  },
  "reward_range": [0.0, 1.0],
  "termination": {
    "terminated": "Environment task terminal condition.",
    "truncated": "Max step or wrapper time limit."
  },
  "public_info_schema": {"type": "dict", "fields": {}}
}
```

Gymnasium spaces map to the common schema: `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`, `Dict`, and `Tuple`. Unknown spaces must be converted to a JSON-safe fallback and marked as non-official unless meanings are supplied.

## WorkspaceContract

```json
{
  "editable_paths": ["system/", "tools/", "experiments/"],
  "readonly_paths": ["AGENTS.md", "task.md", "task_contract.json", "feedback/"],
  "policy_path": "system/policy.py",
  "feedback_layout": {
    "current": "feedback/current/",
    "history": "feedback/history/"
  },
  "allowed_commands": [
    "python -m hlbench.rollout --workspace . --split train --episodes 10 --output-dir experiments/<name>",
    "python -m compileall system/policy.py"
  ],
  "private_data_rules": [
    "validation and heldout seeds are not visible",
    "validation and heldout replays are not generated"
  ]
}
```

## TaskContract

`TaskContract = ScenarioSpec + EnvContract + WorkspaceContract`.

```json
{
  "version": "task-contract-v1",
  "scenario": {},
  "env": {},
  "workspace": {},
  "policy_interface": {
    "class": "Policy",
    "reset": "reset(task_config: dict) -> None",
    "act": "act(observation: Any, context: dict) -> Any"
  },
  "validation_rules": {
    "official_requires_observation_meanings": true,
    "official_requires_action_meanings": true
  }
}
```

`task_contract.json` is the canonical file. `task.md` is rendered from it for the agent.

## AgentConfig

```json
{
  "backend": "command",
  "name": "claude",
  "model_name": "claude-code",
  "command": ["claude"],
  "timeout_seconds": 1800,
  "stdout_path": "submission/stdout.txt",
  "stderr_path": "submission/stderr.txt"
}
```

Codex CLI and Claude Code are command presets (`codex`, `claude`), not special cases in the harness core. Custom agents should use `--agent-backend command --agent-command "..."`.

`model_name` is the filesystem-safe run path segment used in `runs/<model_name>/<env_name>/<run_id>/`.

## Transition

```json
{
  "transition_id": "run_id/epoch_000",
  "run_root": "runs/claude-code/cartpole_balance/example",
  "scenario_id": "cartpole_balance_v0",
  "agent": {"backend": "command", "name": "claude-code"},
  "input": {
    "policy_sha256": "...",
    "train_summary": {}
  },
  "submission": {
    "policy_sha256": "...",
    "policy_path": "epochs/epoch_000/submission/policy.py"
  },
  "result": {
    "policy_status": {
      "compile_ok": true,
      "import_ok": true,
      "missing_policy": false,
      "runtime_failures": []
    },
    "agent_status": {
      "backend": "command",
      "returncode": 0,
      "timed_out": false,
      "final_json_parse_ok": true,
      "agent_failed": false
    },
    "contract_status": {
      "invalid_transition": false,
      "violations": []
    },
    "minimum_score_applied": {
      "applied": false,
      "reason": null
    },
    "summaries": {
      "train": {},
      "validation": {},
      "heldout": {}
    }
  },
  "reward": {
    "total": 0.0,
    "components": {}
  },
  "artifacts": {}
}
```

Transition records must never contain validation or heldout replay details, seeds, or failure traces. Validation and heldout evaluators must not generate replay or per-episode artifacts in the first place.

## ExperimentManifest

```json
{
  "experiment_id": "probe_key_logic",
  "source": "agent",
  "epoch": 0,
  "split": "train",
  "policy_sha256": "...",
  "command": "python -m hlbench.rollout --workspace . --split train --episodes 5 --output-dir experiments/probe_key_logic",
  "episodes": 5,
  "started_at": "2026-05-15T00:00:00Z",
  "ended_at": "2026-05-15T00:00:10Z",
  "artifacts": {
    "summary": "summary.json",
    "failures": "failures.jsonl",
    "replays": "replays/"
  }
}
```

Allowed `source` values are `benchmark`, `agent`, and `diagnostic`. `agent` means an agent-created train-only artifact in `workspace/experiments/`; it does not mean human-facing analysis is visible to the agent.

## ReportMetrics

```json
{
  "run_id": "example",
  "model_name": "claude-code",
  "env_name": "cartpole_balance",
  "scenario_id": "cartpole_balance_v0",
  "epochs": 5,
  "primary": {
    "heldout_return_auc": 0.72,
    "final_heldout_mean_return": 0.91,
    "best_heldout_mean_return": 0.94
  },
  "quality": {
    "invalid_transition_rate": 0.2,
    "minimum_score_count": 1,
    "agent_failure_count": 0,
    "contract_violation_count": 1
  },
  "cost": {
    "train_episodes": 120,
    "agent_wall_time_seconds": 640.5
  }
}
```

Report metrics may include validation and heldout aggregate values, but never private seeds, replays, per-episode records, or failure traces.
