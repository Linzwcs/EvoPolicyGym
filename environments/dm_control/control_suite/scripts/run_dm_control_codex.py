"""Run one explicitly unsafe local Codex development loop on dm_control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import (
    AssessmentConfig,
    ConsoleProgress,
    RunConfig,
    ValidationConfig,
    run,
)

from dm_control_benchmarks import (
    DM_CONTROL_PROFILES,
    DmControlBenchmark,
    DmControlConfig,
    baseline_program,
)


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    max_submissions = (
        namespace.episode_budget
        if namespace.max_submissions is None
        else namespace.max_submissions
    )
    if not namespace.allow_unsafe_process:
        parser.error(
            "local Agent and Policy processes are not isolated; "
            "pass --allow-unsafe-process to acknowledge this"
        )
    record_to = namespace.record_to.resolve()
    if record_to.exists() or record_to.is_symlink():
        parser.error("--record-to must not already exist")
    if not record_to.parent.is_dir():
        parser.error("--record-to parent directory must exist")

    result = run(
        baseline_program(),
        DmControlBenchmark(
            DmControlConfig(
                profile=namespace.profile,
                max_episode_steps=namespace.max_episode_steps,
            )
        ),
        agent=Codex(
            model=namespace.model,
            reasoning_effort=namespace.reasoning_effort,
            executable=namespace.codex_executable,
        ),
        execution=ProcessExecution.unsafe(),
        record_to=record_to,
        config=RunConfig(
            split="train",
            max_submissions=namespace.max_submissions,
            episode_budget=namespace.episode_budget,
            episode_pool_size=namespace.episode_pool_size,
            max_episodes_per_submission=namespace.max_episodes_per_submission,
            validation=ValidationConfig(
                split="validation",
                episodes_per_candidate=namespace.validation_episodes_per_candidate,
                max_candidates=min(2, max_submissions),
            ),
            assessment=AssessmentConfig(
                split="test",
                episodes=namespace.assessment_episodes,
            ),
            seed=namespace.seed,
            episode_timeout_seconds=namespace.episode_timeout_seconds,
            agent_timeout_seconds=namespace.agent_timeout_seconds,
        ),
        observer=ConsoleProgress(mode=namespace.progress),
    )
    print(
        json.dumps(
            {
                "terminal_reason": result.terminal_reason,
                "final_submission_id": result.final_submission_id,
                "record": str(record_to),
                "submissions": [
                    {
                        "submission_id": submission.submission_id,
                        "score": submission.feedback.score,
                        "episodes_used": submission.episodes_used,
                    }
                    for submission in result.submissions
                ],
                "validation_selected": (
                    None
                    if result.validation is None
                    else result.validation.selected_submission_id
                ),
                "assessment_score": (
                    None if result.assessment is None else result.assessment.score
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real Codex Agent against a dm_control Benchmark.",
    )
    parser.add_argument("--record-to", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=DM_CONTROL_PROFILES,
        default="cartpole-swingup",
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--max-episode-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--max-submissions", type=int)
    parser.add_argument("--episode-budget", type=int, default=4)
    parser.add_argument("--episode-pool-size", type=int, default=8)
    parser.add_argument("--max-episodes-per-submission", type=int, default=2)
    parser.add_argument("--validation-episodes-per-candidate", type=int, default=2)
    parser.add_argument("--assessment-episodes", type=int, default=2)
    parser.add_argument("--episode-timeout-seconds", type=float, default=120)
    parser.add_argument("--agent-timeout-seconds", type=float, default=1_800)
    parser.add_argument("--progress", choices=("auto", "plain"), default="plain")
    parser.add_argument("--allow-unsafe-process", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
