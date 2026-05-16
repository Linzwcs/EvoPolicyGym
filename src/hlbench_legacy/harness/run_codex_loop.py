"""Run a persistent-workspace multi-epoch Codex HL loop."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from hlbench.harness.create_real_workspace import create_workspace
from hlbench.harness.run_codex_step import (
    EventLogger,
    PROMPT_VERSION,
    REPO_ROOT,
    _changed_protected_files,
    _compile_policy,
    _compute_reward,
    _copy_train_feedback_to_workspace,
    _evaluate_all_splits,
    _evaluate_split,
    _hash_protected_files,
    _loc,
    _mean_return,
    _policy_patch,
    _run_codex,
    _repo_relative,
    _sha256_text,
    _write_policy_snapshot,
)
from hlbench.rollout.run_policy import Scenario, file_sha256


def run_codex_loop(
    *,
    scenario_name: str = "minigrid_doorkey",
    run_id: str | None = None,
    epochs: int = 5,
    codex_command: str = "codex exec",
    timeout_seconds: int = 1800,
    train_episodes: int | None = 2,
    validation_episodes: int | None = None,
    heldout_episodes: int | None = None,
    min_validation_delta: float = 0.0,
    skip_codex: bool = False,
) -> Path:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if run_id is None:
        run_id = "codex-loop-" + time.strftime("%Y%m%d-%H%M%S")

    workspace = create_workspace(
        run_id=run_id,
        scenario=scenario_name,
        train_episodes=train_episodes,
    )
    run_dir = workspace.parent
    scenario = Scenario.load(scenario_name)
    checkpoints_dir = run_dir / "checkpoints"
    steps_dir = run_dir / "steps"
    checkpoints_dir.mkdir()
    steps_dir.mkdir()
    run_logger = EventLogger(run_dir / "events.jsonl")
    run_logger.log(
        "loop_started",
        scenario=scenario_name,
        workspace=_repo_relative(workspace),
        epochs=epochs,
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        heldout_episodes=heldout_episodes,
        min_validation_delta=min_validation_delta,
    )

    checkpoint_version = 0
    _copy_workspace_snapshot(workspace, checkpoints_dir / "H_000" / "workspace")
    run_logger.log(
        "checkpoint_created",
        checkpoint_version="H_000",
        checkpoint=_repo_relative(checkpoints_dir / "H_000" / "workspace"),
    )
    transitions_path = run_dir / "transitions.jsonl"
    curve: list[dict[str, Any]] = [
        {
            "epoch": -1,
            "accepted_version": "H_000",
            "checkpoint_version": "H_000",
            "checkpoint": "checkpoints/H_000/workspace",
            "mean_return": None,
        }
    ]

    for epoch in range(epochs):
        step_dir = steps_dir / f"epoch_{epoch:03d}"
        step_dir.mkdir()
        epoch_events_path = step_dir / "epoch_events.jsonl"
        event_logger = EventLogger(run_dir / "events.jsonl", epoch_events_path, epoch=epoch)
        event_logger.log(
            "epoch_started",
            accepted_version_before=f"H_{checkpoint_version:03d}",
            checkpoint_version_before=f"H_{checkpoint_version:03d}",
            workspace=_repo_relative(workspace),
        )
        transition = _run_epoch(
            epoch=epoch,
            scenario=scenario,
            scenario_name=scenario_name,
            workspace=workspace,
            step_dir=step_dir,
            codex_command=codex_command,
            timeout_seconds=timeout_seconds,
            train_episodes=train_episodes,
            validation_episodes=validation_episodes,
            heldout_episodes=heldout_episodes,
            min_validation_delta=min_validation_delta,
            skip_codex=skip_codex,
            event_logger=event_logger,
        )

        _copy_workspace_snapshot(workspace, step_dir / "candidate_workspace")
        event_logger.log(
            "candidate_workspace_saved",
            path=_repo_relative(step_dir / "candidate_workspace"),
        )

        checkpoint_version += 1
        checkpoint_dir = checkpoints_dir / f"H_{checkpoint_version:03d}" / "workspace"
        _copy_workspace_snapshot(workspace, checkpoint_dir)
        event_logger.log(
            "checkpoint_created",
            checkpoint_version=f"H_{checkpoint_version:03d}",
            checkpoint=_repo_relative(checkpoint_dir),
            continued_from_candidate=True,
            accepted=transition["result"]["accepted"],
            reject_reason=transition["result"]["reject_reason"],
        )

        transition["result"]["continued"] = True
        transition["result"]["accepted_version_after"] = f"H_{checkpoint_version:03d}"
        transition["result"]["checkpoint_version_after"] = f"H_{checkpoint_version:03d}"
        artifacts = transition["result"].setdefault("artifacts", {})
        artifacts["epoch_events"] = _repo_relative(epoch_events_path)
        artifacts["run_events"] = _repo_relative(run_dir / "events.jsonl")
        artifacts["candidate_workspace"] = _repo_relative(step_dir / "candidate_workspace")
        artifacts["checkpoint_workspace"] = _repo_relative(checkpoint_dir)
        artifacts["transition_json"] = _repo_relative(step_dir / "transition.json")
        (step_dir / "transition.json").write_text(json.dumps(transition, indent=2) + "\n")
        with transitions_path.open("a") as handle:
            handle.write(json.dumps(transition, sort_keys=True) + "\n")
        event_logger.log(
            "transition_written",
            path=_repo_relative(step_dir / "transition.json"),
            transitions_jsonl=_repo_relative(transitions_path),
        )

        heldout_summary = transition["result"]["reward_components"]["summaries_after"]["heldout"]
        curve.append(
            {
                "epoch": epoch,
                "accepted": transition["result"]["accepted"],
                "continued": True,
                "accepted_version": f"H_{checkpoint_version:03d}",
                "checkpoint_version": f"H_{checkpoint_version:03d}",
                "checkpoint": f"checkpoints/H_{checkpoint_version:03d}/workspace",
                "heldout_mean_return": heldout_summary.get("mean_return", 0.0),
                "validation_mean_return": transition["result"]["reward_components"][
                    "summaries_after"
                ]["validation"].get("mean_return", 0.0),
                "train_mean_return": transition["result"]["reward_components"][
                    "summaries_after"
                ]["train"].get("mean_return", 0.0),
            }
        )
        (run_dir / "learning_curve.json").write_text(json.dumps(curve, indent=2) + "\n")
        epoch_summary = {
            "epoch": epoch,
            "accepted": transition["result"]["accepted"],
            "continued": True,
            "reject_reason": transition["result"]["reject_reason"],
            "accepted_version_after": f"H_{checkpoint_version:03d}",
            "checkpoint_version_after": f"H_{checkpoint_version:03d}",
            "reward": transition["reward"],
            "artifacts": artifacts,
        }
        (step_dir / "epoch_summary.json").write_text(json.dumps(epoch_summary, indent=2) + "\n")
        event_logger.log(
            "epoch_completed",
            accepted=transition["result"]["accepted"],
            continued=True,
            reject_reason=transition["result"]["reject_reason"],
            accepted_version_after=f"H_{checkpoint_version:03d}",
            checkpoint_version_after=f"H_{checkpoint_version:03d}",
            reward=transition["reward"],
            epoch_summary=_repo_relative(step_dir / "epoch_summary.json"),
        )

    run_logger.log(
        "loop_completed",
        epochs=epochs,
        accepted_version=f"H_{checkpoint_version:03d}",
        checkpoint_version=f"H_{checkpoint_version:03d}",
    )
    print(run_dir)
    return run_dir


def _run_epoch(
    *,
    epoch: int,
    scenario: Scenario,
    scenario_name: str,
    workspace: Path,
    step_dir: Path,
    codex_command: str,
    timeout_seconds: int,
    train_episodes: int | None,
    validation_episodes: int | None,
    heldout_episodes: int | None,
    min_validation_delta: float,
    skip_codex: bool,
    event_logger: EventLogger,
) -> dict[str, Any]:
    policy_path = workspace / "policy.py"
    before_policy = policy_path.read_text()
    before_sha = file_sha256(policy_path)
    protected_hashes = _hash_protected_files(workspace)
    evaluator_dir = step_dir / "evaluator"

    event_logger.log("train_before_started", episodes=train_episodes)
    before_train = _evaluate_split(
        scenario_name=scenario_name,
        split="train",
        policy_path=policy_path,
        output_dir=evaluator_dir / "before" / "train",
        episodes=train_episodes,
        write_replays=True,
    )
    event_logger.log(
        "train_before_completed",
        run_dir=before_train["run_dir"],
        summary=before_train["summary"],
    )
    before = {"train": before_train}
    _copy_train_feedback_to_workspace(before_train, workspace)
    event_logger.log(
        "train_feedback_copied",
        target=_repo_relative(workspace / "rollout"),
    )

    top_prompt = workspace.parent / "prompt.md"
    prompt_path = step_dir / "prompt.md"
    prompt_path.write_text(
        top_prompt.read_text()
        + f"\n\n## Epoch\n\nThis is outer-loop epoch {epoch}. "
        + "Improve only the current workspace state.\n"
    )
    event_logger.log("prompt_written", path=_repo_relative(prompt_path))

    codex_run = _run_codex(
        workspace=workspace,
        run_dir=step_dir,
        codex_command=codex_command,
        timeout_seconds=timeout_seconds,
        skip_codex=skip_codex,
        event_logger=event_logger,
    )

    after_policy = policy_path.read_text() if policy_path.exists() else ""
    after_sha = _sha256_text(after_policy) if after_policy else ""
    policy_changed = before_sha != after_sha
    protected_changed = _changed_protected_files(workspace, protected_hashes)
    event_logger.log(
        "compile_started",
        policy_changed=policy_changed,
        protected_files_changed=protected_changed,
    )
    compile_ok, compile_error = _compile_policy(policy_path, step_dir)
    event_logger.log(
        "compile_completed",
        compile_ok=compile_ok,
        compile_error=compile_error,
    )

    before_policy_path = _write_policy_snapshot(
        before_policy,
        evaluator_dir / "policies" / "policy_before.py",
    )
    event_logger.log(
        "private_before_eval_started",
        policy_snapshot=_repo_relative(before_policy_path),
    )
    before["validation"] = _evaluate_split(
        scenario_name=scenario_name,
        split="validation",
        policy_path=before_policy_path,
        output_dir=evaluator_dir / "before" / "validation",
        episodes=validation_episodes,
        write_replays=False,
    )
    before["heldout"] = _evaluate_split(
        scenario_name=scenario_name,
        split="heldout",
        policy_path=before_policy_path,
        output_dir=evaluator_dir / "before" / "heldout",
        episodes=heldout_episodes,
        write_replays=False,
    )
    event_logger.log(
        "private_before_eval_completed",
        summaries={
            "validation": before["validation"]["summary"],
            "heldout": before["heldout"]["summary"],
        },
    )

    event_logger.log("after_eval_started")
    after = _evaluate_all_splits(
        scenario_name=scenario_name,
        policy_path=policy_path,
        output_root=evaluator_dir / "after",
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        heldout_episodes=heldout_episodes,
    )
    event_logger.log(
        "after_eval_completed",
        summaries={split: result["summary"] for split, result in after.items()},
    )

    patch = _policy_patch(before_policy, after_policy)
    patch_path = step_dir / "policy.patch"
    patch_path.write_text(patch)
    event_logger.log(
        "policy_patch_written",
        path=_repo_relative(patch_path),
        bytes=patch_path.stat().st_size,
    )

    invalid = (
        not compile_ok
        or bool(protected_changed)
        or codex_run.returncode != 0
        or codex_run.timed_out
    )
    reward = _compute_reward(
        before=before,
        after=after,
        before_loc=_loc(before_policy),
        after_loc=_loc(after_policy),
        invalid=invalid,
    )
    event_logger.log("reward_computed", reward=reward)
    validation_delta = _mean_return(after["validation"]) - _mean_return(before["validation"])
    accepted = (
        policy_changed
        and not invalid
        and validation_delta > min_validation_delta
    )
    reject_reason = ""
    if not accepted:
        reject_reason = _reject_reason(
            policy_changed=policy_changed,
            invalid=invalid,
            validation_delta=validation_delta,
            min_validation_delta=min_validation_delta,
            protected_changed=protected_changed,
            compile_ok=compile_ok,
            codex_returncode=codex_run.returncode,
            codex_timed_out=codex_run.timed_out,
        )
    event_logger.log(
        "candidate_decision",
        accepted=accepted,
        reject_reason=reject_reason,
        validation_delta=validation_delta,
    )

    final_json = codex_run.final_json
    return {
        "transition_id": f"{workspace.parent.name}:epoch_{epoch:03d}",
        "task_id": scenario.scenario_id,
        "sampler": {
            "model": "codex-cli",
            "prompt_version": PROMPT_VERSION,
        },
        "state": {
            "epoch": epoch,
            "policy_before_sha": before_sha,
            "rollout_summary_before": before["train"]["summary"],
            "recent_failures": before["train"]["failures"],
            "previous_patch_outcomes": [],
        },
        "action": {
            "diagnosis": str(final_json.get("summary", "")),
            "edit_plan": [str(item) for item in final_json.get("claimed_improvements", [])],
            "patch": patch,
            "expected_effect": str(final_json.get("next_recommended_check", "")),
            "risk": "\n".join(str(item) for item in final_json.get("known_risks", [])),
        },
        "result": {
            "accepted": accepted,
            "reject_reason": reject_reason,
            "patch_applied": policy_changed,
            "compile_ok": compile_ok,
            "rollout_summary_after": after["train"]["summary"],
            "artifacts": {
                "epoch_events": _repo_relative(step_dir / "epoch_events.jsonl"),
                "run_events": _repo_relative(workspace.parent / "events.jsonl"),
                "prompt": _repo_relative(prompt_path),
                "codex": codex_run.artifacts,
                "policy_patch": _repo_relative(patch_path),
            },
            "reward_components": {
                **reward,
                "policy_after_sha": after_sha,
                "codex_returncode": codex_run.returncode,
                "codex_timed_out": codex_run.timed_out,
                "compile_error": compile_error,
                "protected_files_changed": protected_changed,
                "summaries_before": {
                    split: result["summary"] for split, result in before.items()
                },
                "summaries_after": {
                    split: result["summary"] for split, result in after.items()
                },
            },
        },
        "reward": {
            "total": reward["total"],
            "train_delta": reward["train_delta"],
            "validation_delta": reward["validation_delta"],
            "heldout_delta": reward["heldout_delta"],
            "regression_penalty": reward["regression_penalty"],
            "complexity_penalty": reward["complexity_penalty"],
            "invalid_patch_penalty": reward["invalid_patch_penalty"],
        },
    }


def _reject_reason(
    *,
    policy_changed: bool,
    invalid: bool,
    validation_delta: float,
    min_validation_delta: float,
    protected_changed: list[str],
    compile_ok: bool,
    codex_returncode: int,
    codex_timed_out: bool,
) -> str:
    if not policy_changed:
        return "policy_unchanged"
    if protected_changed:
        return "protected_files_changed"
    if not compile_ok:
        return "compile_failed"
    if codex_timed_out:
        return "codex_timed_out"
    if codex_returncode != 0:
        return "codex_failed"
    if invalid:
        return "invalid_candidate"
    if validation_delta <= min_validation_delta:
        return "validation_not_improved"
    return "not_accepted"


def _copy_workspace_snapshot(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".run_tmp", "rollout", "rollouts", "*.pyc"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="minigrid_doorkey")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--codex-command", default="codex exec")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--heldout-episodes", type=int, default=None)
    parser.add_argument("--min-validation-delta", type=float, default=0.0)
    parser.add_argument("--skip-codex", action="store_true")
    args = parser.parse_args()

    run_codex_loop(
        scenario_name=args.scenario,
        run_id=args.run_id,
        epochs=args.epochs,
        codex_command=args.codex_command,
        timeout_seconds=args.timeout_seconds,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        heldout_episodes=args.heldout_episodes,
        min_validation_delta=args.min_validation_delta,
        skip_codex=args.skip_codex,
    )


if __name__ == "__main__":
    main()
