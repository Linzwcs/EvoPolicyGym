"""Run one explicitly unsafe local Codex development loop on Jumanji."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jumanji_benchmarks import (
    JUMANJI_PROFILES,
    JumanjiBenchmark,
    JumanjiConfig,
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

_DEFAULT_EPISODE_TIMEOUT_SECONDS = 120.0
_PROFILE_EPISODE_TIMEOUT_SECONDS = {
    # Strong 2048 policies routinely survive for 700-1,000 environment steps.
    # On the current JAX backend that can exceed the ordinary 120 second limit.
    "game-2048": 300.0,
    # Snake's success horizon is 4,000 moves, so surviving policies need the
    # same extended allowance instead of being penalized for lasting longer.
    "snake": 300.0,
}


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

    episode_timeout_seconds = namespace.episode_timeout_seconds
    if episode_timeout_seconds is None:
        episode_timeout_seconds = _PROFILE_EPISODE_TIMEOUT_SECONDS.get(
            namespace.profile,
            _DEFAULT_EPISODE_TIMEOUT_SECONDS,
        )

    result = run(
        baseline_program(),
        JumanjiBenchmark(JumanjiConfig(profile=namespace.profile)),
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
            episode_timeout_seconds=episode_timeout_seconds,
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
                "profile": namespace.profile,
                "terminal_reason": result.terminal_reason,
                "final_submission_id": result.final_submission_id,
                "candidate_submission_ids": list(result.candidate_submission_ids),
                "validation": (
                    None
                    if result.validation is None
                    else {
                        "selected_submission_id": result.validation.selected_submission_id,
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
        description="Run a real Codex Agent against one Jumanji profile.",
    )
    parser.add_argument("--profile", choices=JUMANJI_PROFILES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort",
        required=True,
        help="set the selected Codex model's reasoning effort",
    )
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--record-to", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-submissions", type=int)
    parser.add_argument("--episode-budget", type=int, default=128)
    parser.add_argument(
        "--episode-pool-size",
        type=int,
        help="fixed selectable training Episode pool size; defaults to budget",
    )
    parser.add_argument("--max-episodes-per-submission", type=int)
    parser.add_argument(
        "--validation-episodes-per-candidate",
        type=int,
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
        help="enable held-out final-Program Assessment",
    )
    parser.add_argument(
        "--assessment-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        help=(
            "per-Episode timeout; defaults to 300 seconds for game-2048 and "
            "snake, and 120 seconds for other profiles"
        ),
    )
    parser.add_argument("--agent-timeout-seconds", type=float, default=7_200)
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="auto",
        help="render Host-side Run progress to stderr",
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
