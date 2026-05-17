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
    termination_schema = json.dumps(env.termination, indent=2, sort_keys=True)
    context_schema = json.dumps(_context_schema(scenario.max_steps), indent=2, sort_keys=True)
    action_lines = _meaning_lines(scenario.action_meanings)
    observation_lines = _meaning_lines(scenario.observation_meanings)
    notes = _notes_lines(scenario.task.notes)
    success_threshold = _success_threshold_line(scenario.metadata.get("success_score"))
    reward_range = _reward_range_line(env.reward_range)
    return "\n".join(
        [
            "# Task",
            "",
            f"Scenario: `{scenario.scenario_id}`",
            f"Environment: `{env.backend}:{env.env_id}`",
            f"Observation type: `{scenario.observation_type}`",
            f"Max steps: `{scenario.max_steps}`",
            "",
            "## Goal",
            scenario.task.goal,
            notes,
            "",
            "## Environment Loop",
            "The evaluator calls `reset(task_config)` once at the start of each episode, then calls `act(observation, context)` until the episode terminates, truncates, or reaches the max step limit.",
            "You may keep episode-local memory on `self`; clear or initialize it in `reset`.",
            "Evaluator seeds and hidden simulator state are never passed to policy.",
            "",
            "The `context` argument has this public shape:",
            "",
            "```json",
            context_schema,
            "```",
            "",
            "## Success",
            scenario.task.success_condition,
            success_threshold,
            "",
            "## Scoring",
            scenario.task.reward_description,
            reward_range,
            f"Execution errors or invalid actions receive the minimum score `{scenario.minimum_score.value}`.",
            f"Minimum score reason: {scenario.minimum_score.reason}",
            "",
            "## Observation",
            "Policy receives exactly the public `observation` described below. Do not rely on hidden environment state, private seeds, or full maps unless they are present in this schema.",
            "",
            "```json",
            observation_schema,
            "```",
            observation_lines,
            "",
            "## Actions",
            "Policy must return one action matching this schema on every call to `act`.",
            "",
            "```json",
            action_schema,
            "```",
            action_lines,
            "",
            "## Termination",
            "Episode end conditions are reported by the environment as:",
            "",
            "```json",
            termination_schema,
            "```",
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
        label = _meaning_label(item)
        meaning = item.get("meaning", item.get("description", ""))
        lines.append(f"- `{label}`: {meaning}")
    return "\n".join(lines)


def _meaning_label(item: dict[str, Any]) -> str:
    name = item.get("name")
    if "index" in item:
        return f"{item['index']} / {name}" if name else str(item["index"])
    if "id" in item:
        return f"{item['id']} / {name}" if name else str(item["id"])
    if "field" in item:
        return f"{item['field']} / {name}" if name else str(item["field"])
    return str(name or "?")


def _context_schema(max_steps: int) -> dict[str, Any]:
    return {
        "action_schema": "Machine-readable schema for valid return actions.",
        "action_count": "Number of legal discrete actions, or null for continuous action spaces.",
        "step_index": "Zero-based timestep within the current episode.",
        "max_steps": max_steps,
    }


def _success_threshold_line(value: Any) -> str:
    if value is None:
        return ""
    return f"Success threshold used for `success_rate`: score >= `{value}`."


def _reward_range_line(reward_range: tuple[float | None, float | None]) -> str:
    low, high = reward_range
    if low is None and high is None:
        return "Reward range is not bounded by the environment contract."
    return f"Environment reward range: `{low}` to `{high}`."


def _notes_lines(items: tuple[str, ...]) -> str:
    if not items:
        return ""
    lines = ["", "Notes:"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)
