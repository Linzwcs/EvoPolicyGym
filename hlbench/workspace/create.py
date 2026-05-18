"""Create learner workspaces."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from hlbench.core.artifacts import write_json
from hlbench.core.paths import run_root
from hlbench.core.scenario import find_scenario_dir, load_scenario
from hlbench.core.task import WorkspaceContractSpec, build_task_contract, write_task_files
from hlbench.envs.registry import get_backend
from hlbench.workspace.contract import WorkspaceContract


def default_workspace_root(*, scenario_name: str, model_name: str, run_id: str | None = None) -> Path:
    run_name = run_id or time.strftime("%Y%m%d-%H%M%S")
    return run_root(model_name=model_name, env_name=scenario_name, run_id=run_name) / "workspace"


def create_workspace(
    *,
    scenario_name: str,
    run_id: str | None = None,
    model_name: str = "local",
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> WorkspaceContract:
    scenario = load_scenario(scenario_name)
    source = find_scenario_dir(scenario_name)
    root = output_dir or default_workspace_root(scenario_name=scenario.name, model_name=model_name, run_id=run_id)
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for dirname in ("system", "feedback/current", "feedback/history", "tools", "experiments"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "policy.py", root / "system" / "policy.py")

    env_contract = get_backend(scenario.env_backend).describe(scenario)
    task_contract = build_task_contract(
        scenario=scenario,
        env=env_contract,
        workspace=WorkspaceContractSpec(),
    )
    write_task_files(root, task_contract)
    (root / "AGENTS.md").write_text(_render_agents_md(), encoding="utf-8")

    contract = WorkspaceContract(
        root=root,
        scenario_name=scenario.name,
        policy_path=root / "system" / "policy.py",
    )
    write_json(
        contract.manifest_path,
        {
            "scenario": scenario.name,
            "scenario_id": scenario.scenario_id,
            "editable_paths": list(contract.editable_paths),
            "readonly_paths": list(contract.readonly_paths),
            "policy_path": str(contract.policy_path),
        },
    )
    return contract


def _render_agents_md() -> str:
    return """# Workspace Rules

Editable paths:
- system/
- tools/
- experiments/

Read-only paths:
- AGENTS.md
- task.md
- task_contract.json
- feedback/

Rules:
- Implement and keep `system/policy.py` executable.
- Use only heuristic Python policies: controllers, planners, state machines, simple memory, or parameterized rules.
- Do not implement RL training loops, neural-network policies, PPO/SAC/DQN-style updates, learned weight files, or offline RL.
- Rollouts are for debugging, validation, and light heuristic tuning, not for training an RL agent.
- Use `tools/` for reusable analysis helpers, rollout parsers, and small scripts you write while improving the policy.
- Write helper outputs, scratch files, notes, and train-only rollout results under `experiments/`.
- Use `feedback/current/` for the latest train feedback.
- `feedback/history/` may include prior train replays and aggregate validation summaries.
- You may use `feedback/history/*/validation_summary.json` as aggregate validation feedback.
- Do not attempt to inspect validation seeds, validation replays, heldout metrics, or heldout data.
- Do not use `tools/` or `experiments/` to modify the evaluator, task files, feedback, seeds, or hidden data.
"""
