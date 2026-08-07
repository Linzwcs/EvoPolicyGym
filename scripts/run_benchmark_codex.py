"""Run Codex against one explicitly selected no-argument Benchmark factory."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from evopolicygym import Program
from evopolicygym.agents import Codex
from evopolicygym.authoring import Benchmark
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

    benchmark_factory = _load_factory(
        namespace.module,
        namespace.benchmark_factory,
    )
    baseline_factory = _load_factory(
        namespace.module,
        namespace.baseline_factory,
    )
    benchmark = cast(Benchmark, benchmark_factory())
    initial_program = baseline_factory()
    if type(initial_program) is not Program:
        parser.error("the baseline factory must return an exact Program")

    result = run(
        initial_program,
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
                "assessment": (
                    None
                    if result.assessment is None
                    else {
                        "policy_failures": result.assessment.policy_failures,
                        "score": result.assessment.score,
                        "submission_id": result.assessment.submission_id,
                    }
                ),
                "benchmark_id": benchmark.spec.id,
                "candidate_submission_ids": list(result.candidate_submission_ids),
                "final_submission_id": result.final_submission_id,
                "record": str(record_to),
                "submissions": [
                    {
                        "episode_indices": list(submission.episode_indices),
                        "episodes_used": submission.episodes_used,
                        "score": submission.feedback.score,
                        "submission_id": submission.submission_id,
                    }
                    for submission in result.submissions
                ],
                "terminal_reason": result.terminal_reason,
                "validation": (
                    None
                    if result.validation is None
                    else {
                        "candidates": [
                            {
                                "policy_failures": candidate.policy_failures,
                                "score": candidate.score,
                                "submission_id": candidate.submission_id,
                            }
                            for candidate in result.validation.candidates
                        ],
                        "selected_submission_id": (
                            result.validation.selected_submission_id
                        ),
                    }
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_factory(module_name: str, attribute_name: str) -> Callable[[], object]:
    module = importlib.import_module(module_name)
    value = getattr(module, attribute_name, None)
    if not callable(value):
        raise ValueError(
            f"{module_name}.{attribute_name} must be a callable factory"
        )
    return cast(Callable[[], object], value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a real Codex Agent against a Benchmark whose public module "
            "exports no-argument Benchmark and baseline Program factories."
        )
    )
    parser.add_argument("--module", required=True)
    parser.add_argument("--benchmark-factory", required=True)
    parser.add_argument("--baseline-factory", default="baseline_program")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--record-to", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-submissions",
        type=int,
        help="defaults to --episode-budget when omitted",
    )
    parser.add_argument("--episode-budget", type=int, default=12)
    parser.add_argument("--episode-pool-size", type=int)
    parser.add_argument("--max-episodes-per-submission", type=int)
    parser.add_argument("--validation-episodes-per-candidate", type=int)
    parser.add_argument(
        "--validation-split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument("--validation-max-candidates", type=int, default=2)
    parser.add_argument("--assessment-episodes", type=int)
    parser.add_argument(
        "--assessment-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--episode-timeout-seconds", type=float, default=120)
    parser.add_argument("--agent-timeout-seconds", type=float, default=900)
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
