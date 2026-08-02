#!/usr/bin/env python3
"""Summarize public Balatro EvaluationResult or paired-comparison JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--side",
        choices=("auto", "candidate", "baseline", "direct"),
        default="auto",
        help="select a side from comparison JSON; auto prefers candidate",
    )
    parser.add_argument(
        "--win-bonus",
        type=float,
        default=1000.0,
        help="Benchmark win bonus to remove from ordinary progress",
    )
    parser.add_argument(
        "--allow-mixed-digests",
        action="store_true",
        help="permit pooling evidence from more than one Program digest",
    )
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _selected(document: dict[str, Any], side: str) -> dict[str, Any]:
    if side == "direct":
        return document
    if side in {"candidate", "baseline"}:
        return _mapping(document.get(side), side)
    candidate = document.get("candidate")
    return _mapping(candidate, "candidate") if candidate is not None else document


def _reward(episode: dict[str, Any]) -> float | None:
    value = episode.get("reward")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Episode reward must be numeric or null")
    return float(value)


def _episode_index(episode: dict[str, Any]) -> int | None:
    value = episode.get("episode_index")
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("episode_index must be a non-negative integer")
    return value


def _progress(reward: float | None, win_bonus: float) -> tuple[float, bool]:
    if reward is None:
        return 0.0, False
    won = reward >= win_bonus
    return (reward - win_bonus if won else reward), won


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "episodes": len(values),
        "mean_blinds": statistics.fmean(values),
        "median_blinds": statistics.median(values),
        "early_le5": sum(value <= 5 for value in values),
        "ge12": sum(value >= 12 for value in values),
        "ge18": sum(value >= 18 for value in values),
        "max_blinds": max(values),
    }


def main() -> int:
    arguments = _arguments()
    progress: list[float] = []
    raw_rewards: list[float] = []
    failures = 0
    wins = 0
    digests: set[str] = set()
    paired_deltas: list[float] = []

    for path in arguments.paths:
        document = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
        selected = _selected(document, arguments.side)
        digest = selected.get("program_digest")
        if isinstance(digest, str):
            digests.add(digest)
        episodes = selected.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"{path}: selected result has no Episode list")

        selected_progress: list[float] = []
        selected_indices: list[int | None] = []
        for value in episodes:
            episode = _mapping(value, "Episode")
            selected_indices.append(_episode_index(episode))
            reward = _reward(episode)
            item_progress, won = _progress(reward, arguments.win_bonus)
            selected_progress.append(item_progress)
            progress.append(item_progress)
            raw_rewards.append(0.0 if reward is None else reward)
            wins += won
            failures += episode.get("failure") is not None

        compare_candidate = (
            arguments.side in {"auto", "candidate"}
            and "baseline" in document
            and "candidate" in document
        )
        if compare_candidate:
            baseline = _mapping(document["baseline"], "baseline")
            baseline_episodes = baseline.get("episodes")
            if not isinstance(baseline_episodes, list):
                raise ValueError(f"{path}: baseline has no Episode list")
            if len(baseline_episodes) != len(selected_progress):
                raise ValueError(f"{path}: paired Episode counts differ")
            for index, selected_value in enumerate(selected_progress):
                baseline_value = _mapping(
                    baseline_episodes[index],
                    "Episode",
                )
                baseline_index = _episode_index(baseline_value)
                selected_index = selected_indices[index]
                if (
                    (baseline_index is None) != (selected_index is None)
                    or baseline_index != selected_index
                ):
                    raise ValueError(
                        f"{path}: paired Episode indices differ"
                    )
                baseline_reward = _reward(baseline_value)
                baseline_progress, _ = _progress(
                    baseline_reward, arguments.win_bonus
                )
                paired_deltas.append(selected_value - baseline_progress)

    if not progress:
        raise ValueError("no Episodes found")
    if len(digests) > 1 and not arguments.allow_mixed_digests:
        raise ValueError("refusing to pool multiple Program digests")

    output: dict[str, Any] = {
        **_summary(progress),
        "wins": wins,
        "policy_failures": failures,
        "mean_run_reward": statistics.fmean(raw_rewards),
        "program_digests": sorted(digests),
    }
    if paired_deltas:
        output["paired"] = {
            "improved": sum(value > 0 for value in paired_deltas),
            "unchanged": sum(value == 0 for value in paired_deltas),
            "regressed": sum(value < 0 for value in paired_deltas),
            "mean_delta_blinds": statistics.fmean(paired_deltas),
            "min_delta_blinds": min(paired_deltas),
            "max_delta_blinds": max(paired_deltas),
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
