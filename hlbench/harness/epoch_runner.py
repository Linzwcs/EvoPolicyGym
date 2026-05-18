"""Run one harness epoch."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.core.artifacts import write_json
from hlbench.core.complexity import analyze_policy_complexity, complexity_delta
from hlbench.core.events import EventLogger
from hlbench.core.paths import run_root
from hlbench.core.scenario import load_scenario
from hlbench.harness.agents.command import CommandAgent, CommandResult
from hlbench.harness.agents.config import resolve_agent_config
from hlbench.harness.evaluator import (
    CompileResult,
    compile_policy,
    evaluate_split,
    minimum_score_rollout,
)
from hlbench.harness.reward import compute_reward
from hlbench.workspace.contract import WorkspaceContract
from hlbench.workspace.create import create_workspace
from hlbench.workspace.feedback import clear_current_feedback, write_feedback, write_feedback_history


@dataclass(frozen=True)
class EpochResult:
    run_dir: Path
    workspace: Path
    compile_result: CompileResult
    agent_result: CommandResult
    reward: dict[str, Any]
    input: dict[str, Any]
    submission: dict[str, Any]
    reference: dict[str, Any]
    evaluation: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        comparison = {
            "reference": self.reference,
            "reward": self.reward,
        }
        input_complexity = self.input.get("complexity", {})
        submission_complexity = self.submission.get("complexity", {})
        if isinstance(input_complexity, dict) and isinstance(submission_complexity, dict):
            comparison["complexity_delta"] = complexity_delta(input_complexity, submission_complexity)
        return {
            "run_dir": str(self.run_dir),
            "workspace": str(self.workspace),
            "input": self.input,
            "submission": self.submission,
            "evaluation": self.evaluation,
            "comparison": comparison,
        }


def run_epoch(
    *,
    scenario_name: str,
    run_id: str | None = None,
    command: list[str] | None = None,
    agent_backend: str = "command",
    agent_preset: str = "none",
    agent_command: list[str] | None = None,
    model_name: str = "local",
    train_episodes: int | None = 2,
    timeout_seconds: int = 1800,
    workspace_root: Path | None = None,
    epoch_dir: Path | None = None,
    reset_workspace: bool = True,
    prior_feedback: dict[str, Any] | None = None,
) -> EpochResult:
    epoch_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    scenario = load_scenario(scenario_name)
    sampler_seeds = _sampler_seeds(epoch_id)
    workspace = _prepare_workspace(
        scenario_name=scenario_name,
        run_id=epoch_id,
        model_name=model_name,
        workspace_root=workspace_root,
        reset_workspace=reset_workspace,
    )
    run_dir = epoch_dir or workspace.root.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_config = resolve_agent_config(
        backend=agent_backend,
        preset=agent_preset,
        command=agent_command if agent_command is not None else command,
    )
    events = EventLogger(run_dir / "events.jsonl")
    events.log("epoch_started", scenario=scenario_name, workspace=str(workspace.root))

    _publish_current_feedback(workspace=workspace, prior_feedback=prior_feedback)
    readonly_before = _readonly_snapshot(workspace.root)
    input_policy_text = workspace.policy_path.read_text()
    input_policy_sha = _sha256_text(input_policy_text)
    input_complexity = analyze_policy_complexity(workspace.policy_path)
    events.log("agent_started", agent=agent_config.to_record())
    agent_result = CommandAgent(
        command=agent_config.command,
        timeout_seconds=timeout_seconds,
        backend=agent_config.backend,
        name=agent_config.name,
    ).run(workspace.root)
    events.log("agent_completed", agent=agent_result.to_record())

    protected_changed = _readonly_snapshot(workspace.root) != readonly_before
    compile_result = compile_policy(workspace.policy_path)
    events.log("compile_completed", **compile_result.to_record())
    invalid_transition = (not agent_result.ok) or (not compile_result.ok) or protected_changed

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_policy_path = input_dir / "policy.py"
    input_policy_path.write_text(input_policy_text)
    submission_dir = run_dir / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_policy_path = submission_dir / "policy.py"
    submission_policy_path.write_text(_safe_read_text(workspace.policy_path))
    submission_complexity = analyze_policy_complexity(workspace.policy_path)
    agent_record = _write_agent_artifacts(submission_dir, agent_result)
    write_json(submission_dir / "compile.json", compile_result.to_record())
    write_json(input_dir / "complexity.json", input_complexity)
    write_json(submission_dir / "complexity.json", submission_complexity)
    reference = _prior_or_minimum_summaries(prior_feedback=prior_feedback, scenario_name=scenario_name)
    if invalid_transition:
        evaluation = _minimum_transition_splits(
            scenario_name=scenario_name,
            output_root=run_dir / "evaluation",
            train_episodes=train_episodes,
            error=_invalid_reason(
                agent_ok=agent_result.ok,
                compile_ok=compile_result.ok,
                protected_changed=protected_changed,
            ),
        )
    else:
        train_evaluation = evaluate_split(
            scenario_name=scenario_name,
            split="train",
            policy_path=workspace.policy_path,
            output_dir=run_dir / "evaluation" / "train",
            episodes=train_episodes,
            sampler_seed=sampler_seeds["train"],
        )
        evaluation = {
            "train": train_evaluation,
            **_evaluate_hidden_splits(
                scenario_name=scenario_name,
                policy_path=workspace.policy_path,
                output_root=run_dir / "evaluation",
            ),
        }
    reward = compute_reward(
        reference={key: _summary(value) for key, value in reference.items()},
        evaluation={key: value.summary for key, value in evaluation.items()},
        compile_ok=compile_result.ok,
        agent_ok=agent_result.ok,
        evaluation_ok=not _evaluation_has_minimum_score(evaluation),
        minimum_score=scenario.minimum_score.value,
        protected_changed=protected_changed,
    )
    write_feedback(
        workspace,
        summary=evaluation["train"].summary,
        failures=evaluation["train"].failures,
        source_run_dir=evaluation["train"].run_dir,
    )
    write_feedback_history(
        workspace,
        epoch_id=_history_epoch_id(run_dir=run_dir, fallback=epoch_id),
        train_summary=evaluation["train"].summary,
        train_failures=evaluation["train"].failures,
        train_source_run_dir=evaluation["train"].run_dir,
        validation_summary=evaluation["validation"].summary,
    )
    submission_policy_sha = _safe_file_sha256(workspace.policy_path)
    result = EpochResult(
        run_dir=run_dir,
        workspace=workspace.root,
        compile_result=compile_result,
        agent_result=agent_result,
        reward=reward,
        input={
            "policy_path": str(input_policy_path),
            "policy_sha256": input_policy_sha,
            "complexity": input_complexity,
            "feedback_source": _feedback_source(prior_feedback),
        },
        submission={
            "policy_path": str(submission_policy_path),
            "policy_sha256": submission_policy_sha,
            "complexity": submission_complexity,
            "compile": compile_result.to_record(),
            "agent": agent_record,
            "protected_changed": protected_changed,
        },
        reference=reference,
        evaluation={key: value.to_record() for key, value in evaluation.items()},
    )
    transition_record = result.to_record()
    write_json(run_dir / "transition.json", transition_record)
    write_json(run_dir / "epoch.json", transition_record)
    events.log("epoch_completed", reward=reward)
    return result


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _safe_file_sha256(path: Path) -> str | None:
    return _file_sha256(path) if path.exists() else None


def _summary(record: Any) -> dict[str, Any]:
    if hasattr(record, "summary"):
        return dict(record.summary)
    return dict(record.get("summary", {}))


def _history_epoch_id(*, run_dir: Path, fallback: str) -> str:
    return run_dir.name if run_dir.name.startswith("epoch_") else fallback


def _write_agent_artifacts(submission_dir: Path, result: CommandResult) -> dict[str, object]:
    stdout_path = submission_dir / "stdout.txt"
    stderr_path = submission_dir / "stderr.txt"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)
    record = result.to_record(stdout_path="stdout.txt", stderr_path="stderr.txt")
    write_json(submission_dir / "agent.json", record)
    return record


def _feedback_source(prior_feedback: dict[str, Any] | None) -> dict[str, Any]:
    if prior_feedback is None:
        return {"kind": "none", "reason": "epoch_0_no_initial_policy_run"}
    train = prior_feedback.get("train", {})
    return {
        "kind": "previous_epoch_train_feedback",
        "train_summary": train.get("summary", {}),
    }


def _publish_current_feedback(
    *,
    workspace: WorkspaceContract,
    prior_feedback: dict[str, Any] | None,
) -> None:
    if prior_feedback is None:
        clear_current_feedback(workspace)
        return
    train = prior_feedback["train"]
    write_feedback(
        workspace,
        summary=train["summary"],
        failures=train.get("failures", []),
        source_run_dir=Path(str(train["run_dir"])),
    )


def _prior_or_minimum_summaries(
    *,
    prior_feedback: dict[str, Any] | None,
    scenario_name: str,
) -> dict[str, Any]:
    if prior_feedback is not None:
        return prior_feedback
    scenario = load_scenario(scenario_name)
    minimum_summary = {
        "episodes": 0,
        "successes": 0,
        "success_rate": 0.0,
        "mean_score": scenario.minimum_score.value,
        "mean_steps": 0.0,
        "invalid_actions": 0,
        "minimum_score_episodes": 0,
    }
    return {
        "train": {"summary": dict(minimum_summary), "trials": [], "failures": []},
        "validation": {"summary": dict(minimum_summary), "trials": [], "failures": []},
        "heldout": {"summary": dict(minimum_summary), "trials": [], "failures": []},
    }


def _readonly_snapshot(root: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in [root / "AGENTS.md", root / "task.md", root / "task_contract.json"]:
        if path.exists():
            snapshot[str(path.relative_to(root))] = _file_sha256(path)
    feedback = root / "feedback"
    if feedback.exists():
        for path in sorted(item for item in feedback.rglob("*") if item.is_file()):
            snapshot[str(path.relative_to(root))] = _file_sha256(path)
    return snapshot


def _evaluate_hidden_splits(
    *,
    scenario_name: str,
    policy_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    return {
        "validation": evaluate_split(
            scenario_name=scenario_name,
            split="validation",
            policy_path=policy_path,
            output_dir=output_root / "validation",
            episodes=None,
            sampler_seed=None,
        ),
        "heldout": evaluate_split(
            scenario_name=scenario_name,
            split="heldout",
            policy_path=policy_path,
            output_dir=output_root / "heldout",
            episodes=None,
            sampler_seed=None,
        ),
    }


def _evaluation_has_minimum_score(evaluation: dict[str, Any]) -> bool:
    return any(
        int(result.summary.get("minimum_score_episodes", 0)) > 0
        for result in evaluation.values()
    )


def _minimum_transition_splits(
    *,
    scenario_name: str,
    output_root: Path,
    train_episodes: int | None,
    error: str,
) -> dict[str, Any]:
    return {
        "train": minimum_score_rollout(
            scenario_name=scenario_name,
            split="train",
            output_dir=output_root / "train",
            episodes=train_episodes,
            error=error,
        ),
        "validation": minimum_score_rollout(
            scenario_name=scenario_name,
            split="validation",
            output_dir=output_root / "validation",
            episodes=None,
            error=error,
        ),
        "heldout": minimum_score_rollout(
            scenario_name=scenario_name,
            split="heldout",
            output_dir=output_root / "heldout",
            episodes=None,
            error=error,
        ),
    }


def _invalid_reason(*, agent_ok: bool, compile_ok: bool, protected_changed: bool) -> str:
    reasons: list[str] = []
    if not agent_ok:
        reasons.append("agent_command_failed")
    if not compile_ok:
        reasons.append("policy_compile_failed")
    if protected_changed:
        reasons.append("workspace_readonly_path_changed")
    return ",".join(reasons)


def _sampler_seeds(epoch_id: str) -> dict[str, int]:
    digest = hashlib.sha256(epoch_id.encode()).digest()
    base = int.from_bytes(digest[:8], "big")
    return {
        "train": base,
        "validation": base + 1,
        "heldout": base + 2,
    }


def _prepare_workspace(
    *,
    scenario_name: str,
    run_id: str,
    model_name: str,
    workspace_root: Path | None,
    reset_workspace: bool,
) -> WorkspaceContract:
    if workspace_root is None:
        return create_workspace(
            scenario_name=scenario_name,
            run_id=run_id,
            model_name=model_name,
            overwrite=True,
        )
    if reset_workspace or not workspace_root.exists():
        return create_workspace(
            scenario_name=scenario_name,
            output_dir=workspace_root,
            overwrite=reset_workspace,
        )
    return WorkspaceContract(
        root=workspace_root,
        scenario_name=scenario_name,
        policy_path=workspace_root / "system" / "policy.py",
    )


def default_loop_root(*, scenario_name: str, model_name: str, run_id: str) -> Path:
    return run_root(model_name=model_name, env_name=scenario_name, run_id=run_id)
