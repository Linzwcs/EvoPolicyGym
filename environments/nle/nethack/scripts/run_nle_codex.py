"""Run one bounded Core16-style NLE Codex policy-evolution loop."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from evopolicygym import AssessmentConfig, RunConfig, ValidationConfig, run
from evopolicygym.agents import Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import ConsoleProgress

from nle_benchmarks import NetHackBenchmark, NetHackConfig, baseline_program
from nle_benchmarks.diagnostics import identical_validation_feedback_groups
from nle_benchmarks.evidence import MAX_PUBLIC_FEEDBACK_EPISODES

_AGENT_TOOL_MODULES = ("numpy", "PIL", "imageio.v3", "imageio_ffmpeg")


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if not namespace.allow_unsafe_process:
        parser.error(
            "local Agent and Policy processes are not isolated by "
            "EvoPolicyGym; pass --allow-unsafe-process to acknowledge this"
        )
    if not 1 <= namespace.max_episodes_per_submission <= MAX_PUBLIC_FEEDBACK_EPISODES:
        parser.error(
            "--max-episodes-per-submission must be between 1 and "
            f"{MAX_PUBLIC_FEEDBACK_EPISODES} for bounded complete Feedback"
        )
    _require_agent_tools(parser)

    record_to = namespace.record_to.resolve()
    if record_to.exists() or record_to.is_symlink():
        parser.error("--record-to must not already exist")
    if not record_to.parent.is_dir():
        parser.error("--record-to parent directory must exist")
    session_socket = record_to / "control" / "session.sock"
    if len(os.fsencode(session_socket)) >= 108:
        parser.error(
            "--record-to is too long for the Linux Unix-domain Session socket; "
            "choose a shorter absolute path (for example under /data/tmp)"
        )

    benchmark = NetHackBenchmark(
        NetHackConfig(max_episode_steps=namespace.max_episode_steps)
    )
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
            executable=namespace.codex_executable,
            view_image=True,
        ),
        execution=ProcessExecution.unsafe(),
        record_to=record_to,
        config=RunConfig(
            split="train",
            max_submissions=namespace.max_submissions,
            episode_budget=namespace.episode_budget,
            max_episodes_per_submission=namespace.max_episodes_per_submission,
            bulk_feedback_retention_bytes=namespace.bulk_feedback_retention_bytes,
            use_benchmark_skill=namespace.benchmark_skill,
            validation=ValidationConfig(
                split="validation",
                episodes_per_candidate=namespace.validation_episodes_per_candidate,
                max_candidates=namespace.validation_max_candidates,
            ),
            assessment=AssessmentConfig(
                split="test",
                episodes=namespace.assessment_episodes,
            ),
            seed=namespace.seed,
            episode_timeout_seconds=namespace.episode_timeout_seconds,
            agent_timeout_seconds=namespace.agent_timeout_seconds,
        ),
        observer=observer,
    )
    identical_feedback_groups = identical_validation_feedback_groups(
        result.validation
    )

    print(
        json.dumps(
            {
                "terminal_reason": result.terminal_reason,
                "final_submission_id": result.final_submission_id,
                "candidate_submission_ids": list(result.candidate_submission_ids),
                "identical_validation_feedback_groups": (
                    identical_feedback_groups
                ),
                "record": str(record_to),
                "agent_visual_tools": {
                    "view_image": True,
                    "python_modules": list(_AGENT_TOOL_MODULES),
                    "environment_generated_visualizations": False,
                },
                "submissions": [
                    {
                        "submission_id": submission.submission_id,
                        "program_digest": submission.program_digest,
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
                                "program_digest": candidate.program_digest,
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
                        "program_digest": result.assessment.program_digest,
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


def _require_agent_tools(parser: argparse.ArgumentParser) -> None:
    missing: list[str] = []
    for module in _AGENT_TOOL_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        parser.error(
            "missing Agent analysis dependencies "
            f"{', '.join(missing)}; run `uv sync --extra agent-tools`"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Core16-style NLE experiment with complete raw training "
            "evidence and Agent-controlled analysis."
        )
    )
    parser.add_argument("--record-to", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--max-episode-steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--max-submissions", type=int, default=32)
    parser.add_argument("--episode-budget", type=int, default=1_024)
    parser.add_argument("--max-episodes-per-submission", type=int, default=64)
    parser.add_argument("--validation-episodes-per-candidate", type=int, default=64)
    parser.add_argument("--validation-max-candidates", type=int, default=3)
    parser.add_argument("--assessment-episodes", type=int, default=256)
    parser.add_argument("--episode-timeout-seconds", type=float, default=600)
    parser.add_argument("--agent-timeout-seconds", type=float, default=172_800)
    parser.add_argument(
        "--bulk-feedback-retention-bytes",
        type=int,
        default=1_024 * 1_024 * 1_024,
        help="combined Host and Agent bulk retention capacity",
    )
    parser.add_argument(
        "--benchmark-skill",
        action="store_true",
        help="publish the packaged Benchmark skill (disabled by default)",
    )
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
