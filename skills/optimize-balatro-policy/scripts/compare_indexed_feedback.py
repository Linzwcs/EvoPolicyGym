#!/usr/bin/env python3
"""Compare indexed Balatro feedback files on exactly matched Episode indices."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        type=Path,
        help="baseline feedback.json; repeat for multiple files",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=Path,
        help="candidate feedback.json; repeat for multiple files",
    )
    parser.add_argument("--win-bonus", type=float, default=1000.0)
    return parser.parse_args()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _outcome(
    episode: dict[str, Any],
    *,
    win_bonus: float,
) -> tuple[float, float, bool, bool]:
    reward = episode.get("reward")
    failed = episode.get("failure") is not None
    if reward is None:
        if not failed:
            raise ValueError("null reward requires a Policy failure")
        return 0.0, 0.0, False, True
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise ValueError("Episode reward must be numeric or null")
    raw = float(reward)
    won = raw >= win_bonus
    progress = raw - win_bonus if won else raw
    return progress, raw, won, failed


def _load(
    paths: list[Path],
    *,
    win_bonus: float,
) -> tuple[str, dict[int, list[tuple[float, float, bool, bool]]]]:
    digests: set[str] = set()
    indexed: dict[int, list[tuple[float, float, bool, bool]]] = {}
    for path in paths:
        document = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            str(path),
        )
        digest = document.get("program_digest")
        if not isinstance(digest, str):
            raise ValueError(f"{path}: missing Program digest")
        digests.add(digest)
        episodes = document.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"{path}: missing Episode list")
        for value in episodes:
            episode = _mapping(value, "Episode")
            index = episode.get("episode_index")
            if type(index) is not int or index < 0:
                raise ValueError(f"{path}: invalid episode_index")
            indexed.setdefault(index, []).append(
                _outcome(episode, win_bonus=win_bonus)
            )
    if len(digests) != 1:
        raise ValueError("each comparison side must contain one Program digest")
    return next(iter(digests)), indexed


def _summary(
    outcomes: list[tuple[float, float, bool, bool]],
) -> dict[str, float | int]:
    progress = [item[0] for item in outcomes]
    rewards = [item[1] for item in outcomes]
    return {
        "evaluations": len(outcomes),
        "mean_blinds": statistics.fmean(progress),
        "median_blinds": statistics.median(progress),
        "early_le5": sum(value <= 5 for value in progress),
        "ge12": sum(value >= 12 for value in progress),
        "ge18": sum(value >= 18 for value in progress),
        "wins": sum(item[2] for item in outcomes),
        "policy_failures": sum(item[3] for item in outcomes),
        "mean_run_reward": statistics.fmean(rewards),
    }


def main() -> int:
    arguments = _arguments()
    baseline_digest, baseline = _load(
        arguments.baseline,
        win_bonus=arguments.win_bonus,
    )
    candidate_digest, candidate = _load(
        arguments.candidate,
        win_bonus=arguments.win_bonus,
    )
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate Episode index sets differ")

    baseline_outcomes: list[tuple[float, float, bool, bool]] = []
    candidate_outcomes: list[tuple[float, float, bool, bool]] = []
    deltas: list[float] = []
    for index in sorted(baseline):
        baseline_items = baseline[index]
        candidate_items = candidate[index]
        if len(baseline_items) != len(candidate_items):
            raise ValueError(
                f"Episode index {index} has different repeat counts"
            )
        for offset, baseline_item in enumerate(baseline_items):
            candidate_item = candidate_items[offset]
            baseline_outcomes.append(baseline_item)
            candidate_outcomes.append(candidate_item)
            deltas.append(candidate_item[0] - baseline_item[0])

    output = {
        "baseline": {
            "program_digest": baseline_digest,
            **_summary(baseline_outcomes),
        },
        "candidate": {
            "program_digest": candidate_digest,
            **_summary(candidate_outcomes),
        },
        "matched": {
            "unique_episode_indices": len(baseline),
            "evaluations": len(deltas),
            "improved": sum(value > 0 for value in deltas),
            "unchanged": sum(value == 0 for value in deltas),
            "regressed": sum(value < 0 for value in deltas),
            "mean_delta_blinds": statistics.fmean(deltas),
            "median_delta_blinds": statistics.median(deltas),
            "min_delta_blinds": min(deltas),
            "max_delta_blinds": max(deltas),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
