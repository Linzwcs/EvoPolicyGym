"""Run one protocol-conforming local Codex optimization loop on Crafter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crafter_benchmarks import (
    CrafterBenchmark,
    CrafterConfig,
    CrafterLongHorizonBenchmark,
    CrafterSurvivalDevelopmentBenchmark,
    baseline_program,
)

from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import (
    AssessmentConfig,
    ConsoleProgress,
    RunConfig,
    ValidationConfig,
    run,
)
from evopolicygym.skills import AgentSkill


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if not namespace.allow_unsafe_process:
        parser.error(
            "local Agent and Policy processes are not isolated by "
            "EvoPolicyGym; pass --allow-unsafe-process to acknowledge this"
        )

    record_to = namespace.record_to.resolve()
    if record_to.exists() or record_to.is_symlink():
        parser.error("--record-to must not already exist")
    if not record_to.parent.is_dir():
        parser.error("--record-to parent directory must exist")

    config = CrafterConfig(max_episode_steps=namespace.max_episode_steps)
    benchmark_types = {
        "canonical": CrafterBenchmark,
        "long-horizon": CrafterLongHorizonBenchmark,
        "survival-development": CrafterSurvivalDevelopmentBenchmark,
    }
    benchmark = benchmark_types[namespace.profile](config)
    observer = (
        None
        if namespace.progress == "off"
        else ConsoleProgress(mode=namespace.progress)
    )

    result = run(
        baseline_program(),
        benchmark,
        agent=Codex(
            model=namespace.model,
            reasoning_effort=namespace.reasoning_effort,
            executable=namespace.codex_executable,
            view_image=True,
        ),
        execution=ProcessExecution.unsafe(),
        record_to=record_to,
        config=RunConfig(
            split=namespace.split,
            max_submissions=namespace.max_submissions,
            episode_budget=namespace.episode_budget,
            max_episodes_per_submission=namespace.max_episodes_per_submission,
            bulk_feedback_retention_bytes=(
                namespace.bulk_feedback_retention_bytes
            ),
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
        skills=(
            (
                AgentSkill.from_directory(
                    Path(__file__).parents[1]
                    / "environments"
                    / "crafter"
                    / "crafter"
                    / "skills"
                    / "optimize-crafter-policy"
                ),
            )
            if namespace.benchmark_skill
            else ()
        ),
        observer=observer,
    )

    print(
        json.dumps(
            {
                "terminal_reason": result.terminal_reason,
                "final_submission_id": result.final_submission_id,
                "candidate_submission_ids": list(
                    result.candidate_submission_ids
                ),
                "record": str(record_to),
                "submissions": [
                    {
                        "submission_id": submission.submission_id,
                        "score": submission.feedback.score,
                        "episodes_used": submission.episodes_used,
                    }
                    for submission in result.submissions
                ],
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
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Codex against the Crafter EvoPolicyGym Benchmark."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--record-to", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("survival-development", "long-horizon", "canonical"),
        default="survival-development",
    )
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-submissions", type=int, default=40)
    parser.add_argument("--episode-budget", type=int, default=512)
    parser.add_argument("--max-episodes-per-submission", type=int, default=16)
    parser.add_argument(
        "--bulk-feedback-retention-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help=(
            "combined Host and Agent-workspace capacity for old bulk "
            "Artifacts; the newest submission is always protected"
        ),
    )
    parser.add_argument(
        "--benchmark-skill",
        action="store_true",
        help="publish the packaged Benchmark skill (disabled by default)",
    )
    parser.add_argument(
        "--validation-episodes-per-candidate",
        type=int,
        default=32,
        help="enable post-Agent server-side Validation",
    )
    parser.add_argument(
        "--validation-split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--validation-max-candidates", type=int, default=3)
    parser.add_argument(
        "--assessment-episodes",
        type=int,
        default=32,
        help="enable held-out final-Policy Assessment",
    )
    parser.add_argument(
        "--assessment-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--episode-timeout-seconds", type=float, default=600)
    parser.add_argument("--agent-timeout-seconds", type=float, default=43_200)
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="auto",
    )
    parser.add_argument(
        "--allow-unsafe-process",
        action="store_true",
        help=(
            "acknowledge that EvoPolicyGym ProcessExecution is not a sandbox; "
            "use a caller-owned Codex wrapper for stronger Agent isolation"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
