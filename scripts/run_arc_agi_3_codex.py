"""Run one explicitly unsafe local Codex development loop on ARC-AGI-3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc_agi_3_benchmarks import ArcAgi3Benchmark, ArcAgi3Config, baseline_program

from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import (
    AssessmentConfig,
    ConsoleProgress,
    RunConfig,
    ValidationConfig,
    run,
)


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if not namespace.allow_unsafe_process:
        parser.error(
            "local Agent and Policy processes are not isolated; "
            "pass --allow-unsafe-process to acknowledge this"
        )

    record_to = Path(namespace.record_to)
    if record_to.exists() or record_to.is_symlink():
        parser.error("--record-to must not already exist")
    if not record_to.parent.is_dir():
        parser.error("--record-to parent directory must exist")

    game_ids = tuple(namespace.game_id)
    benchmark = ArcAgi3Benchmark(
        ArcAgi3Config(
            profile="custom" if game_ids else "public-25",
            custom_game_ids=game_ids,
            max_episode_steps=namespace.max_episode_steps,
        ),
        environments_dir=namespace.environments_dir,
        recordings_dir=namespace.recordings_dir,
    )
    result = run(
        baseline_program(),
        benchmark,
        agent=Codex(
            model=namespace.model,
            reasoning_effort=namespace.reasoning_effort,
            executable=namespace.codex_executable,
        ),
        execution=ProcessExecution.unsafe(),
        record_to=record_to,
        config=RunConfig(
            split=namespace.split,
            max_submissions=namespace.max_submissions,
            episode_budget=namespace.episode_budget,
            episode_pool_size=namespace.episode_pool_size,
            max_episodes_per_submission=namespace.max_episodes_per_submission,
            validation=(
                None
                if namespace.validation_episodes_per_candidate is None
                else ValidationConfig(
                    split=namespace.validation_split,
                    episodes_per_candidate=(
                        namespace.validation_episodes_per_candidate
                    ),
                    max_candidates=namespace.validation_max_candidates,
                )
            ),
            assessment=(
                None
                if namespace.assessment_episodes is None
                else AssessmentConfig(
                    split=namespace.assessment_split,
                    episodes=namespace.assessment_episodes,
                )
            ),
            seed=namespace.seed,
            episode_timeout_seconds=namespace.episode_timeout_seconds,
            agent_timeout_seconds=namespace.agent_timeout_seconds,
        ),
        observer=(
            None
            if namespace.progress == "off"
            else ConsoleProgress(mode=namespace.progress)
        ),
    )
    print(
        json.dumps(
            {
                "terminal_reason": result.terminal_reason,
                "final_submission_id": result.final_submission_id,
                "candidate_submission_ids": list(result.candidate_submission_ids),
                "validation": (
                    None
                    if result.validation is None
                    else {
                        "selected_submission_id": (
                            result.validation.selected_submission_id
                        ),
                        "candidates": [
                            {
                                "submission_id": candidate.submission_id,
                                "score": candidate.score,
                                "policy_failures": candidate.policy_failures,
                            }
                            for candidate in result.validation.candidates
                        ],
                    }
                ),
                "assessment": (
                    None
                    if result.assessment is None
                    else {
                        "submission_id": result.assessment.submission_id,
                        "score": result.assessment.score,
                        "policy_failures": result.assessment.policy_failures,
                    }
                ),
                "record": str(record_to),
                "submissions": [
                    {
                        "submission_id": submission.submission_id,
                        "score": submission.feedback.score,
                        "episode_indices": list(submission.episode_indices),
                        "episodes_used": submission.episodes_used,
                    }
                    for submission in result.submissions
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real Codex Agent against the ARC-AGI-3 Benchmark.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--record-to", type=Path, required=True)
    parser.add_argument(
        "--game-id",
        action="append",
        default=[],
        metavar="GAME_ID",
        help=(
            "select one versioned game ID; repeat for a custom collection; "
            "omit to use public-25"
        ),
    )
    parser.add_argument(
        "--environments-dir",
        default="runs/arc-agi-3/environments",
    )
    parser.add_argument(
        "--recordings-dir",
        default="runs/arc-agi-3/recordings",
    )
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-submissions", type=int, default=16)
    parser.add_argument("--episode-budget", type=int, default=1_024)
    parser.add_argument("--episode-pool-size", type=int)
    parser.add_argument("--max-episodes-per-submission", type=int)
    parser.add_argument("--validation-episodes-per-candidate", type=int)
    parser.add_argument(
        "--validation-split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--validation-max-candidates", type=int, default=3)
    parser.add_argument("--assessment-episodes", type=int)
    parser.add_argument(
        "--assessment-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--episode-timeout-seconds", type=float, default=60)
    parser.add_argument("--agent-timeout-seconds", type=float, default=3_600)
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="auto",
    )
    parser.add_argument(
        "--allow-unsafe-process",
        action="store_true",
        help=(
            "acknowledge that the Agent and submitted Policy run with the "
            "current user's authority"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
