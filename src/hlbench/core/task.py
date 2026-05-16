"""Task and policy contracts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from hlbench.core.artifacts import write_json
from hlbench.core.scenario import ScenarioSpec


@runtime_checkable
class PolicyProtocol(Protocol):
    def act(self, observation: Any, context: dict[str, Any]) -> Any:
        """Return any action object accepted by the active environment backend."""


@dataclass(frozen=True)
class EnvContract:
    backend: str
    env_id: str
    observation_schema: dict[str, Any]
    action_schema: dict[str, Any]
    reward_range: tuple[float | None, float | None]
    termination: dict[str, Any]
    public_info_schema: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceContractSpec:
    editable_paths: tuple[str, ...] = ("system/", "tools/", "experiments/")
    readonly_paths: tuple[str, ...] = ("AGENTS.md", "task.md", "task_contract.json", "feedback/")
    policy_path: str = "system/policy.py"
    allowed_commands: tuple[str, ...] = (
        "python -m compileall system",
        "python -m hlbench.rollout.cli --workspace . --split train --episodes 10 --output-dir experiments/<name>",
    )
    private_data_rules: tuple[str, ...] = (
        "Train feedback is visible inside feedback/current and feedback/history.",
        "Aggregate validation summaries may appear in feedback/history.",
        "Validation seeds, rollouts, replays, per-episode records, and failures are never exposed.",
        "Heldout data and metrics are never exposed inside the workspace.",
        "Do not modify task.md, task_contract.json, AGENTS.md, or feedback/.",
    )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskContract:
    version: str
    scenario: ScenarioSpec
    env: EnvContract
    workspace: WorkspaceContractSpec
    policy_interface: dict[str, str]
    validation_rules: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.scenario.scenario_id

    def to_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "scenario": self.scenario.to_record(include_private_splits=False),
            "env": self.env.to_record(),
            "workspace": self.workspace.to_record(),
            "policy_interface": self.policy_interface,
            "validation_rules": self.validation_rules,
            "metadata": self.metadata,
        }


def build_task_contract(
    *,
    scenario: ScenarioSpec,
    env: EnvContract,
    workspace: WorkspaceContractSpec | None = None,
) -> TaskContract:
    return TaskContract(
        version="0.1",
        scenario=scenario,
        env=env,
        workspace=workspace or WorkspaceContractSpec(),
        policy_interface={
            "class": "Policy",
            "reset": "Optional reset(task_config: dict) -> None",
            "act": "Required act(observation: Any, context: dict) -> Any",
        },
        validation_rules={
            "action_validation": "Returned actions must match action_schema.",
            "minimum_score": scenario.minimum_score.value,
            "minimum_score_applies_to": [
                "policy import errors",
                "reset or act exceptions",
                "invalid actions",
                "workspace contract violations",
                "evaluator execution failures",
            ],
        },
    )


def render_task_markdown(contract: TaskContract) -> str:
    scenario = contract.scenario
    env = contract.env
    workspace = contract.workspace
    action_schema = json.dumps(env.action_schema, indent=2, sort_keys=True)
    observation_schema = json.dumps(env.observation_schema, indent=2, sort_keys=True)
    action_lines = _meaning_lines(scenario.action_meanings)
    observation_lines = _meaning_lines(scenario.observation_meanings)
    notes = _notes_lines(scenario.task.notes)
    return "\n".join(
        [
            "# Task",
            "",
            f"Scenario: `{scenario.scenario_id}`",
            f"Environment: `{env.backend}:{env.env_id}`",
            f"Max steps: `{scenario.max_steps}`",
            "",
            "## Goal",
            scenario.task.goal,
            notes,
            "",
            "## Success",
            scenario.task.success_condition,
            "",
            "## Scoring",
            scenario.task.reward_description,
            f"Execution errors or invalid actions receive the minimum score `{scenario.minimum_score.value}`.",
            "",
            "## Observation",
            "Policy receives only the public observation described by this schema:",
            "",
            "```json",
            observation_schema,
            "```",
            observation_lines,
            "",
            "## Actions",
            "Policy must return an action matching this schema:",
            "",
            "```json",
            action_schema,
            "```",
            action_lines,
            "",
            "## Policy Interface",
            f"Implement `{workspace.policy_path}` with `class Policy`.",
            "`reset(task_config: dict)` is optional. Real evaluator seeds are not passed to policy.",
            "`act(observation, context)` is required and must return one valid action.",
            "",
            "## Workspace Rules",
            "Editable paths: " + ", ".join(f"`{item}`" for item in workspace.editable_paths),
            "Read-only paths: " + ", ".join(f"`{item}`" for item in workspace.readonly_paths),
            "Feedback contains train rollouts plus optional aggregate validation history. Heldout data never appears in `feedback/`.",
            "",
            "## Train Command",
            "```bash",
            "python -m hlbench.rollout.cli --workspace . --split train --episodes 10 --output-dir experiments/<name>",
            "```",
            "",
        ]
    )


def write_task_files(root: Path, contract: TaskContract) -> None:
    write_json(root / "task_contract.json", contract.to_record())
    (root / "task.md").write_text(render_task_markdown(contract), encoding="utf-8")


def _meaning_lines(items: tuple[dict[str, Any], ...]) -> str:
    if not items:
        return ""
    lines = ["", "Meanings:"]
    for item in items:
        label = item.get("name", item.get("id", item.get("index", "?")))
        meaning = item.get("meaning", item.get("description", ""))
        lines.append(f"- `{label}`: {meaning}")
    return "\n".join(lines)


def _notes_lines(items: tuple[str, ...]) -> str:
    if not items:
        return ""
    lines = ["", "Notes:"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)
