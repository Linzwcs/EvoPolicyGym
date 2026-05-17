"""Write public feedback into learner workspaces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from hlbench.core.artifacts import write_json, write_jsonl
from hlbench.workspace.contract import WorkspaceContract


def write_feedback(
    workspace: WorkspaceContract | Path,
    *,
    summary: dict[str, Any],
    failures: list[dict[str, Any]] | None = None,
    source_run_dir: Path | None = None,
) -> Path:
    root = workspace.root if isinstance(workspace, WorkspaceContract) else workspace
    feedback_dir = root / "feedback" / "current"
    _reset_dir(feedback_dir)
    _write_train_feedback(
        feedback_dir,
        summary=summary,
        failures=failures,
        source_run_dir=source_run_dir,
    )
    return feedback_dir


def clear_current_feedback(workspace: WorkspaceContract | Path) -> Path:
    root = workspace.root if isinstance(workspace, WorkspaceContract) else workspace
    feedback_dir = root / "feedback" / "current"
    _reset_dir(feedback_dir)
    write_json(
        feedback_dir / "manifest.json",
        {
            "source": "none",
            "reason": "no_prior_evaluation",
        },
    )
    return feedback_dir


def write_feedback_history(
    workspace: WorkspaceContract | Path,
    *,
    epoch_id: str,
    train_summary: dict[str, Any],
    train_failures: list[dict[str, Any]] | None = None,
    train_source_run_dir: Path | None = None,
    validation_summary: dict[str, Any] | None = None,
) -> Path:
    root = workspace.root if isinstance(workspace, WorkspaceContract) else workspace
    history_dir = root / "feedback" / "history" / epoch_id
    _reset_dir(history_dir)
    train_dir = history_dir / "train"
    _write_train_feedback(
        train_dir,
        summary=train_summary,
        failures=train_failures,
        source_run_dir=train_source_run_dir,
    )
    if validation_summary is not None:
        write_json(history_dir / "validation_summary.json", validation_summary)
    write_json(
        history_dir / "manifest.json",
        {
            "epoch_id": epoch_id,
            "contains_train_replays": (train_source_run_dir / "replays").exists()
            if train_source_run_dir is not None
            else False,
            "contains_validation_aggregate": validation_summary is not None,
        },
    )
    return history_dir


def _write_train_feedback(
    feedback_dir: Path,
    *,
    summary: dict[str, Any],
    failures: list[dict[str, Any]] | None,
    source_run_dir: Path | None,
) -> None:
    feedback_dir.mkdir(parents=True, exist_ok=True)
    write_json(feedback_dir / "summary.json", summary)
    write_jsonl(feedback_dir / "failures.jsonl", failures or [])
    write_json(
        feedback_dir / "manifest.json",
        {
            "split": "train",
            "source": "benchmark_train_rollout",
        },
    )
    if source_run_dir is not None:
        episodes_path = source_run_dir / "episodes.jsonl"
        replays_dir = source_run_dir / "replays"
        if episodes_path.exists():
            _copy_episode_records(episodes_path, feedback_dir / "episodes.jsonl")
        if replays_dir.exists():
            target_replays = feedback_dir / "replays"
            if target_replays.exists():
                shutil.rmtree(target_replays)
            shutil.copytree(replays_dir, target_replays)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_episode_records(source: Path, target: Path) -> None:
    records: list[dict[str, Any]] = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        replay_path = record.get("replay_path")
        if replay_path:
            record["replay_path"] = f"replays/{Path(str(replay_path)).name}"
        records.append(record)
    write_jsonl(target, records)
