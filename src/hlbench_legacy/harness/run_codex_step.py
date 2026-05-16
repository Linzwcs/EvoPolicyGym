"""Run one automated Codex heuristic-learning step."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import py_compile
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.harness.create_real_workspace import create_workspace
from hlbench.rollout.run_policy import Scenario, file_sha256, run_policy


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
PROMPT_VERSION = "codex-harness-step-v0"


@dataclass(frozen=True)
class CodexRun:
    returncode: int
    timed_out: bool
    final_json: dict[str, Any]
    artifacts: dict[str, Any]


class EventLogger:
    def __init__(self, *paths: Path, epoch: int | None = None) -> None:
        self.paths = tuple(paths)
        self.epoch = epoch

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "time": _utc_timestamp(),
            "event": event,
        }
        if self.epoch is not None:
            record["epoch"] = self.epoch
        record.update(payload)
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record


def run_codex_step(
    *,
    scenario_name: str = "minigrid_doorkey",
    run_id: str | None = None,
    codex_command: str = "codex exec",
    timeout_seconds: int = 1800,
    train_episodes: int | None = 2,
    validation_episodes: int | None = None,
    heldout_episodes: int | None = None,
    skip_codex: bool = False,
) -> Path:
    workspace = create_workspace(
        run_id=run_id,
        scenario=scenario_name,
        train_episodes=train_episodes,
    )
    run_dir = workspace.parent
    event_logger = EventLogger(run_dir / "events.jsonl")
    event_logger.log(
        "step_started",
        scenario=scenario_name,
        workspace=_repo_relative(workspace),
        train_episodes=train_episodes,
        validation_episodes=validation_episodes,
        heldout_episodes=heldout_episodes,
    )
    scenario = Scenario.load(scenario_name)
    policy_path = workspace / "policy.py"

    before_policy = policy_path.read_text()
    before_sha = file_sha256(policy_path)
    protected_hashes = _hash_protected_files(workspace)

    evaluator_dir = run_dir / "evaluator"
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

    codex_run = _run_codex(
        workspace=workspace,
        run_dir=run_dir,
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
    compile_ok, compile_error = _compile_policy(policy_path, run_dir)
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
    patch_path = run_dir / "policy.patch"
    patch_path.write_text(patch)
    event_logger.log(
        "policy_patch_written",
        path=_repo_relative(patch_path),
        bytes=patch_path.stat().st_size,
    )

    reward = _compute_reward(
        before=before,
        after=after,
        before_loc=_loc(before_policy),
        after_loc=_loc(after_policy),
        invalid=(
            not compile_ok
            or bool(protected_changed)
            or codex_run.returncode != 0
            or codex_run.timed_out
        ),
    )
    event_logger.log("reward_computed", reward=reward)

    final_json = codex_run.final_json
    transition = {
        "transition_id": run_dir.name,
        "task_id": scenario.scenario_id,
        "sampler": {
            "model": "codex-cli",
            "prompt_version": PROMPT_VERSION,
        },
        "state": {
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
            "patch_applied": policy_changed,
            "compile_ok": compile_ok,
            "rollout_summary_after": after["train"]["summary"],
            "artifacts": {
                "run_events": _repo_relative(run_dir / "events.jsonl"),
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

    transition["result"]["artifacts"]["transition_json"] = _repo_relative(
        run_dir / "transition.json"
    )
    (run_dir / "transition.json").write_text(json.dumps(transition, indent=2) + "\n")
    event_logger.log(
        "transition_written",
        path=_repo_relative(run_dir / "transition.json"),
    )
    event_logger.log("step_completed")
    print(run_dir)
    return run_dir


def _run_codex(
    *,
    workspace: Path,
    run_dir: Path,
    codex_command: str,
    timeout_seconds: int,
    skip_codex: bool,
    event_logger: EventLogger | None = None,
) -> CodexRun:
    codex_dir = run_dir / "codex"
    codex_dir.mkdir()
    prompt_path = run_dir / "prompt.md"
    prompt = prompt_path.read_text()
    cmd = shlex.split(codex_command)
    started_at = _utc_timestamp()
    start_monotonic = time.monotonic()
    if event_logger is not None:
        event_logger.log(
            "codex_started",
            command=cmd,
            cwd=_repo_relative(workspace),
            prompt_path=_repo_relative(prompt_path),
            timeout_seconds=timeout_seconds,
            skip_codex=skip_codex,
        )

    if skip_codex:
        final = {
            "status": "partial",
            "edited_files": [],
            "commands_run": [],
            "train_rollouts_run": 0,
            "policy_changed": False,
            "summary": "Codex execution skipped by orchestrator.",
            "claimed_improvements": [],
            "known_risks": ["No learner step was run."],
        }
        (codex_dir / "stdout.txt").write_text("")
        (codex_dir / "stderr.txt").write_text("")
        (codex_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
        artifacts = _write_codex_run_metadata(
            codex_dir=codex_dir,
            command=cmd,
            cwd=workspace,
            prompt_path=prompt_path,
            started_at=started_at,
            elapsed_seconds=time.monotonic() - start_monotonic,
            returncode=0,
            timed_out=False,
            final_json_parse_ok=True,
        )
        if event_logger is not None:
            event_logger.log(
                "codex_completed",
                returncode=0,
                timed_out=False,
                elapsed_seconds=artifacts["elapsed_seconds"],
                artifacts=artifacts,
            )
        return CodexRun(returncode=0, timed_out=False, final_json=final, artifacts=artifacts)

    try:
        completed = subprocess.run(
            cmd,
            cwd=workspace,
            input=prompt,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = _text_or_empty(exc.stdout)
        stderr = _text_or_empty(exc.stderr) + f"\nTimed out after {timeout_seconds}s.\n"
        returncode = -1
        timed_out = True

    (codex_dir / "stdout.txt").write_text(stdout)
    (codex_dir / "stderr.txt").write_text(stderr)
    final = _extract_harness_json(stdout)
    final_json_parse_ok = final is not None
    if final is None:
        final = {
            "status": "failed" if returncode else "partial",
            "edited_files": [],
            "commands_run": [],
            "train_rollouts_run": 0,
            "policy_changed": False,
            "summary": "Codex did not return parseable final harness JSON.",
            "claimed_improvements": [],
            "known_risks": ["Final JSON could not be parsed from stdout."],
        }
    (codex_dir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    artifacts = _write_codex_run_metadata(
        codex_dir=codex_dir,
        command=cmd,
        cwd=workspace,
        prompt_path=prompt_path,
        started_at=started_at,
        elapsed_seconds=time.monotonic() - start_monotonic,
        returncode=returncode,
        timed_out=timed_out,
        final_json_parse_ok=final_json_parse_ok,
    )
    if event_logger is not None:
        event_logger.log(
            "codex_completed",
            returncode=returncode,
            timed_out=timed_out,
            elapsed_seconds=artifacts["elapsed_seconds"],
            artifacts=artifacts,
        )
    return CodexRun(
        returncode=returncode,
        timed_out=timed_out,
        final_json=final,
        artifacts=artifacts,
    )


def _write_codex_run_metadata(
    *,
    codex_dir: Path,
    command: list[str],
    cwd: Path,
    prompt_path: Path,
    started_at: str,
    elapsed_seconds: float,
    returncode: int,
    timed_out: bool,
    final_json_parse_ok: bool,
) -> dict[str, Any]:
    stdout_path = codex_dir / "stdout.txt"
    stderr_path = codex_dir / "stderr.txt"
    final_path = codex_dir / "final.json"
    run_path = codex_dir / "run.json"
    ended_at = _utc_timestamp()
    metadata: dict[str, Any] = {
        "command": command,
        "cwd": _repo_relative(cwd),
        "prompt_path": _repo_relative(prompt_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "returncode": returncode,
        "timed_out": timed_out,
        "final_json_parse_ok": final_json_parse_ok,
        "stdout_path": _repo_relative(stdout_path),
        "stderr_path": _repo_relative(stderr_path),
        "final_json_path": _repo_relative(final_path),
        "run_metadata_path": _repo_relative(run_path),
        "stdout_bytes": stdout_path.stat().st_size if stdout_path.exists() else 0,
        "stderr_bytes": stderr_path.stat().st_size if stderr_path.exists() else 0,
    }
    run_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _evaluate_all_splits(
    *,
    scenario_name: str,
    policy_path: Path,
    output_root: Path,
    train_episodes: int | None,
    validation_episodes: int | None,
    heldout_episodes: int | None,
    train_replays: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        "train": _evaluate_split(
            scenario_name=scenario_name,
            split="train",
            policy_path=policy_path,
            output_dir=output_root / "train",
            episodes=train_episodes,
            write_replays=train_replays,
        ),
        "validation": _evaluate_split(
            scenario_name=scenario_name,
            split="validation",
            policy_path=policy_path,
            output_dir=output_root / "validation",
            episodes=validation_episodes,
            write_replays=False,
        ),
        "heldout": _evaluate_split(
            scenario_name=scenario_name,
            split="heldout",
            policy_path=policy_path,
            output_dir=output_root / "heldout",
            episodes=heldout_episodes,
            write_replays=False,
        ),
    }


def _evaluate_split(
    *,
    scenario_name: str,
    split: str,
    policy_path: Path,
    output_dir: Path,
    episodes: int | None,
    write_replays: bool = True,
) -> dict[str, Any]:
    try:
        run_dir = run_policy(
            scenario_name=scenario_name,
            split=split,
            policy_path=policy_path,
            episodes=episodes,
            run_id=None,
            output_dir=output_dir,
            write_replays=write_replays,
        )
        return {
            "ok": True,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "summary": _load_json(run_dir / "rollout" / "summary.json"),
            "failures": _load_jsonl(run_dir / "rollout" / "failures.jsonl"),
        }
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = _error_summary(split, repr(exc))
        (output_dir / "error.json").write_text(json.dumps(summary, indent=2) + "\n")
        return {
            "ok": False,
            "run_dir": str(output_dir.relative_to(REPO_ROOT)),
            "summary": summary,
            "failures": [{"cluster": "evaluation_error", "summary": repr(exc), "count": 1}],
        }


def _copy_train_feedback_to_workspace(train_result: dict[str, Any], workspace: Path) -> None:
    source = REPO_ROOT / train_result["run_dir"] / "rollout"
    target = workspace / "rollout"
    if target.exists():
        shutil.rmtree(target)
    if source.exists():
        shutil.copytree(source, target)


def _write_policy_snapshot(policy_text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(policy_text)
    return path


def _compile_policy(policy_path: Path, run_dir: Path) -> tuple[bool, str]:
    checks_dir = run_dir / "checks"
    checks_dir.mkdir(exist_ok=True)
    try:
        py_compile.compile(str(policy_path), doraise=True)
    except Exception as exc:
        message = repr(exc)
        (checks_dir / "compile_error.txt").write_text(message + "\n")
        return False, message
    (checks_dir / "compile_ok.txt").write_text("ok\n")
    return True, ""


def _compute_reward(
    *,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    before_loc: int,
    after_loc: int,
    invalid: bool,
) -> dict[str, float]:
    train_delta = _mean_return(after["train"]) - _mean_return(before["train"])
    validation_delta = _mean_return(after["validation"]) - _mean_return(before["validation"])
    heldout_delta = _mean_return(after["heldout"]) - _mean_return(before["heldout"])
    regression_penalty = max(0.0, -validation_delta) + max(0.0, -heldout_delta)
    complexity_penalty = 0.0001 * max(0, after_loc - before_loc)
    invalid_patch_penalty = 1.0 if invalid else 0.0
    total = (
        heldout_delta
        + 0.25 * validation_delta
        + 0.1 * train_delta
        - regression_penalty
        - complexity_penalty
        - invalid_patch_penalty
    )
    return {
        "total": total,
        "train_delta": train_delta,
        "validation_delta": validation_delta,
        "heldout_delta": heldout_delta,
        "regression_penalty": regression_penalty,
        "complexity_penalty": complexity_penalty,
        "invalid_patch_penalty": invalid_patch_penalty,
    }


def _extract_harness_json(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    match: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if _looks_like_harness_result(candidate):
            match = candidate
    return match


def _looks_like_harness_result(candidate: Any) -> bool:
    return (
        isinstance(candidate, dict)
        and isinstance(candidate.get("status"), str)
        and isinstance(candidate.get("edited_files"), list)
        and isinstance(candidate.get("commands_run"), list)
    )


def _policy_patch(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="policy_before.py",
            tofile="policy_after.py",
        )
    )


def _hash_protected_files(workspace: Path) -> dict[str, str]:
    protected = [workspace / "task_spec.md", workspace / "tools" / "run_rollout.py"]
    return {str(path.relative_to(workspace)): file_sha256(path) for path in protected}


def _changed_protected_files(workspace: Path, before: dict[str, str]) -> list[str]:
    changed = []
    for relative, previous_hash in before.items():
        path = workspace / relative
        if not path.exists() or file_sha256(path) != previous_hash:
            changed.append(relative)
    return changed


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _mean_return(result: dict[str, Any]) -> float:
    return float(result["summary"].get("mean_return", 0.0))


def _error_summary(split: str, error: str) -> dict[str, Any]:
    return {
        "split": split,
        "episodes": 0,
        "success_rate": 0.0,
        "mean_return": 0.0,
        "mean_steps": 0.0,
        "terminated": 0,
        "truncated": 0,
        "invalid_action_episodes": 0,
        "failure_modes": [{"cluster": "evaluation_error", "summary": error, "count": 1}],
    }


def _loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _text_or_empty(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="minigrid_doorkey")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--codex-command", default="codex exec")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--train-episodes", type=int, default=2)
    parser.add_argument("--validation-episodes", type=int, default=None)
    parser.add_argument("--heldout-episodes", type=int, default=None)
    parser.add_argument("--skip-codex", action="store_true")
    args = parser.parse_args()

    run_codex_step(
        scenario_name=args.scenario,
        run_id=args.run_id,
        codex_command=args.codex_command,
        timeout_seconds=args.timeout_seconds,
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        heldout_episodes=args.heldout_episodes,
        skip_codex=args.skip_codex,
    )


if __name__ == "__main__":
    main()
