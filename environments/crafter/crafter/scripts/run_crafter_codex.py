"""Run one protocol-conforming local Codex optimization loop on Crafter."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from evopolicygym import Program
from evopolicygym.agents import AgentInvocation, AgentTask, Codex
from evopolicygym.execution import ProcessExecution
from evopolicygym.run import (
    AssessmentConfig,
    ConsoleProgress,
    RunConfig,
    ValidationConfig,
    run,
)
from evopolicygym.skills import AgentSkill

from crafter_benchmarks import (
    CrafterBenchmark,
    CrafterConfig,
    CrafterLongHorizonSurvivalBenchmark,
    baseline_program,
    local_symbolic_baseline_program,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CrafterCodex:
    """Add caller-owned Crafter Run conditions to Codex."""

    model: str
    reasoning_effort: str
    executable: str
    observation_profile: str = "rgb"

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        base = Codex(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            executable=self.executable,
            view_image=self.observation_profile == "rgb",
        ).build_invocation(task)
        instructions = [
            _player_guide_instruction(),
            _agent_analysis_python_instruction(self.observation_profile),
            _training_episode_diversity_instruction(),
        ]
        instruction = "\n".join(instructions)
        option = (
            "-c",
            f"developer_instructions={json.dumps(instruction)}",
        )
        identity = dict(base.identity)
        identity["agent_python_tools"] = (
            "numpy"
            if self.observation_profile == "local-symbolic-v1"
            else "numpy,pillow"
        )
        identity["max_train_index_uses"] = "1"
        identity["crafter_observation_profile"] = self.observation_profile
        return AgentInvocation(
            command=(*base.command[:-1], *option, base.command[-1]),
            recorded_command=(
                *base.recorded_command[:-1],
                *option,
                base.recorded_command[-1],
            ),
            identity=identity,
            instructions=base.instructions,
            inherited_environment=base.inherited_environment,
            stdout_media_type=base.stdout_media_type,
        )


def _player_guide_instruction() -> str:
    return """\
Before the first submission, read program/PLAYER_GUIDE.md in full. Treat it as
the authoritative gameplay-mechanics reference available for this Run,
including the observation layout, survival systems, terrain collision,
creatures, gathering, crafting, placement, and day-night behavior. Treat the
Benchmark public specification as the authoritative evaluation objective and
scoring contract. Policy design must jointly respect the game mechanics in
PLAYER_GUIDE.md and the objective in the Benchmark specification; neither
source should be ignored in favor of the other. PLAYER_GUIDE.md provides
environment knowledge, not a prescribed Policy or fixed Action plan. You
remain responsible for discovering and evaluating the strategy.
"""


def _agent_analysis_python_instruction(observation_profile: str) -> str:
    if observation_profile == "local-symbolic-v1":
        return """\
Use `python` directly for Crafter feedback analysis. NumPy is available for
loading the lossless local-symbolic NPZ observations. Do not invoke `uv run`
from the Run workspace; it selects a different project environment.
"""
    return """\
Use `python` directly for Crafter feedback analysis. NumPy and Pillow are
available for loading NPZ observations and producing images. Do not invoke
`uv run` from the Run workspace; it selects a different project environment.
"""


def _training_episode_diversity_instruction() -> str:
    return """\
Training Episode diversity is an explicit requirement for this Run. Each
Run-local training Episode index may be selected at most once across the entire
Run. Never select an index that has already been evaluated. Every new
submission must use previously unseen indices. Keep a simple index-usage record
under analysis/ and check it before every submission. This requirement does not
prescribe a submission size or require full budget consumption.
"""


def _starting_program(observation_profile: str) -> Program:
    if observation_profile == "rgb":
        return baseline_program()
    if observation_profile == "local-symbolic-v1":
        return local_symbolic_baseline_program()
    raise ValueError("Crafter observation profile is invalid")


def _benchmark_skill_directory(observation_profile: str) -> Path:
    if observation_profile == "rgb":
        name = "optimize-crafter-policy"
    elif observation_profile == "local-symbolic-v1":
        name = "optimize-crafter-local-symbolic-policy"
    else:
        raise ValueError("Crafter observation profile is invalid")
    return Path(__file__).parents[1] / "skills" / name


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
    session_socket = record_to / "control" / "session.sock"
    if len(os.fsencode(session_socket)) >= 108:
        parser.error(
            "--record-to is too long for the Linux Unix-domain Session socket; "
            "choose a shorter path under runs/"
        )

    config = CrafterConfig(
        max_episode_steps=namespace.max_episode_steps,
        include_mp4_feedback=namespace.include_mp4_feedback,
        observation_profile=namespace.observation_profile,
    )
    benchmark_types: dict[str, type[CrafterBenchmark]] = {
        "canonical": CrafterBenchmark,
        "lhs": (
            CrafterLongHorizonSurvivalBenchmark
        ),
    }
    benchmark = benchmark_types[namespace.profile](config)
    observer = (
        None
        if namespace.progress == "off"
        else ConsoleProgress(mode=namespace.progress)
    )

    result = run(
        _starting_program(namespace.observation_profile),
        benchmark,
        agent=_CrafterCodex(
            model=namespace.model,
            reasoning_effort=namespace.reasoning_effort,
            executable=namespace.codex_executable,
            observation_profile=namespace.observation_profile,
        ),
        execution=ProcessExecution.unsafe(),
        record_to=record_to,
        config=RunConfig(
            split=namespace.split,
            max_submissions=namespace.max_submissions,
            episode_budget=namespace.episode_budget,
            max_episodes_per_submission=namespace.max_episodes_per_submission,
            finish_budget_policy=namespace.finish_budget_policy,
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
                    _benchmark_skill_directory(namespace.observation_profile)
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
        choices=(
            "lhs",
            "canonical",
        ),
        default="lhs",
        help=(
            "select the default long-horizon LHS metric or upstream "
            "canonical Crafter scoring (default: lhs)"
        ),
    )
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
    parser.add_argument(
        "--observation-profile",
        choices=("rgb", "local-symbolic-v1"),
        default="rgb",
        help=(
            "select canonical RGB or the separate local-symbolic-v1 Policy "
            "observation task (default: rgb)"
        ),
    )
    parser.add_argument(
        "--include-mp4-feedback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "also publish one derived H.264 MP4 replay per train Episode; "
            "lossless NPZ observations remain enabled and authoritative "
            "(default: disabled)"
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-submissions", type=int, default=1024)
    parser.add_argument("--episode-budget", type=int, default=1024)
    parser.add_argument("--max-episodes-per-submission", type=int, default=64)
    parser.add_argument(
        "--finish-budget-policy",
        choices=("allow_early", "require_budget_exhaustion"),
        default="allow_early",
        help=(
            "allow an Agent to finish before exhausting the training Episode "
            "budget (default), or explicitly require complete consumption"
        ),
    )
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
        default=256,
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
        default=512,
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
            "stronger isolation requires caller-owned whole-Run virtualization"
        ),
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
