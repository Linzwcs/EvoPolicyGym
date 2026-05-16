"""Rollout summary helpers."""

from __future__ import annotations

from typing import Any


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(trials)
    successes = sum(1 for trial in trials if trial.get("success"))
    scores = [float(trial.get("score", 0.0)) for trial in trials]
    steps = [int(trial.get("steps", 0)) for trial in trials]
    invalid = sum(int(trial.get("invalid_actions", 0)) for trial in trials)
    minimum_score_count = sum(1 for trial in trials if trial.get("minimum_score_applied"))
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count if count else 0.0,
        "mean_score": sum(scores) / count if count else 0.0,
        "mean_steps": sum(steps) / count if count else 0.0,
        "invalid_actions": invalid,
        "minimum_score_episodes": minimum_score_count,
    }


def failure_samples(trials: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for trial in trials:
        if trial.get("success"):
            continue
        record = {
            "score": trial.get("score"),
            "steps": trial.get("steps"),
            "terminated_by": trial.get("terminated_by"),
            "exception": trial.get("exception"),
            "minimum_score_applied": trial.get("minimum_score_applied", False),
        }
        if "seed" in trial:
            record["seed"] = trial["seed"]
        if "episode_id" in trial:
            record["episode_id"] = trial["episode_id"]
        failures.append(record)
        if len(failures) >= limit:
            break
    return failures
