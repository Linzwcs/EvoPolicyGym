"""Summarize rollout trial records."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def _failure_mode(trial: dict[str, Any]) -> str | None:
    if trial.get("terminated_by") == "success":
        return None
    if trial.get("terminated_by") == "policy_exception":
        return "policy_exception"
    if trial.get("terminated_by") == "invalid_action":
        return "invalid_action"
    if trial.get("terminated_by") == "timeout":
        histogram = trial.get("action_histogram", {})
        if len(histogram) <= 1:
            return "repeats_one_action_until_timeout"
        if float(trial.get("score", 0.0)) <= 0:
            return "no_reward_progress"
        return "timed_out"
    return "never_reached_goal"


def summarize_trials(
    *,
    scenario_id: str,
    env_id: str,
    split: str,
    trials: list[dict[str, Any]],
    policy_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    successes = [trial for trial in trials if trial.get("terminated_by") == "success"]
    scores = [float(trial.get("score", 0.0)) for trial in trials]
    steps = [int(trial.get("steps", 0)) for trial in trials]
    modes = Counter(mode for trial in trials if (mode := _failure_mode(trial)))

    failure_records = [
        {
            "failure_id": f"failure_{index:06d}",
            "cluster": mode,
            "count": count,
            "summary": _failure_summary(mode, count),
        }
        for index, (mode, count) in enumerate(modes.most_common())
    ]

    summary = {
        "scenario_id": scenario_id,
        "env_id": env_id,
        "split": split,
        "policy_sha256": policy_sha256,
        "episodes": len(trials),
        "success_rate": (len(successes) / len(trials)) if trials else 0.0,
        "mean_return": mean(scores) if scores else 0.0,
        "mean_steps": mean(steps) if steps else 0.0,
        "terminated": sum(1 for trial in trials if trial.get("terminated")),
        "truncated": sum(1 for trial in trials if trial.get("truncated")),
        "invalid_action_episodes": sum(1 for trial in trials if trial.get("invalid_actions", 0) > 0),
        "failure_modes": failure_records,
    }
    return summary, failure_records


def _failure_summary(mode: str, count: int) -> str:
    messages = {
        "policy_exception": "Policy raised an exception during rollout.",
        "invalid_action": "Policy produced an invalid action.",
        "repeats_one_action_until_timeout": "Policy repeated one action and timed out.",
        "no_reward_progress": "Policy did not obtain reward before timeout.",
        "timed_out": "Policy timed out before reaching the goal.",
        "never_reached_goal": "Policy did not reach the goal.",
    }
    return f"{messages.get(mode, 'Unclassified rollout failure')} Episodes: {count}."

