#!/usr/bin/env python3
"""Evaluate two Balatro Programs on the same public Benchmark schedule."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from balatro import BalatroBenchmark  # type: ignore[import-not-found]

from evopolicygym import EvaluationConfig, Program, evaluate
from evopolicygym.execution import ProcessExecution
from evopolicygym.results import EvaluationResult


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-program",
        type=Path,
        default=repository
        / (
            "runs/balatro-skill-ab-gpt-5.6-sol-retry-20260725-164423/"
            "workspace/program"
        ),
    )
    parser.add_argument(
        "--candidate-program",
        type=Path,
        default=repository / "human-in-loop/balatro/program",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_726)
    parser.add_argument("--episode-timeout-seconds", type=float, default=60)
    parser.add_argument(
        "--allow-unsafe-process",
        action="store_true",
        help="acknowledge that local Policy subprocesses are not isolated",
    )
    arguments = parser.parse_args(argv)
    if not arguments.allow_unsafe_process:
        parser.error(
            "ProcessExecution is not a sandbox; pass --allow-unsafe-process"
        )
    if arguments.output.exists() or arguments.output.is_symlink():
        parser.error("--output must not already exist")
    if not arguments.output.parent.is_dir():
        parser.error("--output parent must exist")
    return arguments


def _json_value(result: EvaluationResult) -> dict[str, Any]:
    return {
        "benchmark_id": result.benchmark_id,
        "environment_digest": result.environment_digest,
        "program_digest": result.program_digest,
        "feedback": {
            "score": result.feedback.score,
            "content": result.feedback.content,
        },
        "episodes": [asdict(episode) for episode in result.episodes],
    }


def _write_result(directory: Path, result: EvaluationResult) -> None:
    directory.mkdir()
    (directory / "feedback.json").write_text(
        json.dumps(
            _json_value(result),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for artifact in result.feedback.artifacts:
        (directory / artifact.name).write_bytes(artifact.read_bytes())


def _reward(summary: object) -> float:
    reward = getattr(summary, "reward", None)
    return float(reward) if isinstance(reward, (int, float)) else 0.0


def _comparison(
    *,
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    split: str,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    baseline_rewards = [_reward(item) for item in baseline.episodes]
    candidate_rewards = [_reward(item) for item in candidate.episodes]
    return {
        "schema": "evopolicygym-balatro/human-paired-evaluation-v1",
        "split": split,
        "episodes": episodes,
        "seed": seed,
        "baseline": _json_value(baseline),
        "candidate": _json_value(candidate),
        "paired_reward_delta": [
            candidate_reward - baseline_reward
            for baseline_reward, candidate_reward in zip(
                baseline_rewards,
                candidate_rewards,
                strict=True,
            )
        ],
        "mean_score_delta": (
            candidate.feedback.score - baseline.feedback.score
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    config = EvaluationConfig(
        split=arguments.split,
        episodes=arguments.episodes,
        seed=arguments.seed,
        episode_timeout_seconds=arguments.episode_timeout_seconds,
    )
    execution = ProcessExecution.unsafe()

    print(
        f"evaluating baseline: {arguments.episodes} "
        f"{arguments.split} Episodes",
        flush=True,
    )
    baseline = evaluate(
        Program.from_directory(arguments.baseline_program),
        BalatroBenchmark(),
        execution=execution,
        config=config,
    )
    print(
        f"baseline score={baseline.feedback.score:.3f}",
        flush=True,
    )

    print(
        f"evaluating candidate: {arguments.episodes} "
        f"{arguments.split} Episodes",
        flush=True,
    )
    candidate = evaluate(
        Program.from_directory(arguments.candidate_program),
        BalatroBenchmark(),
        execution=execution,
        config=config,
    )
    print(
        f"candidate score={candidate.feedback.score:.3f}",
        flush=True,
    )

    arguments.output.mkdir()
    _write_result(arguments.output / "baseline", baseline)
    _write_result(arguments.output / "candidate", candidate)
    comparison = _comparison(
        baseline=baseline,
        candidate=candidate,
        split=arguments.split,
        episodes=arguments.episodes,
        seed=arguments.seed,
    )
    (arguments.output / "comparison.json").write_text(
        json.dumps(
            comparison,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"mean delta={comparison['mean_score_delta']:+.3f}; "
        f"results={arguments.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
