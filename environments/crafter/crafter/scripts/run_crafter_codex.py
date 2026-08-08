"""Run one protocol-conforming local Codex optimization loop on Crafter."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

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
    CrafterCanonicalStrongSurvivalRepeatBenchmark,
    CrafterCanonicalSurvivalBenchmark,
    CrafterCanonicalSurvivalRepeatBenchmark,
    CrafterConfig,
    CrafterLongHorizonBenchmark,
    CrafterSurvivalDevelopmentBenchmark,
    baseline_program,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _CrafterCodex:
    """Add caller-owned Crafter Run conditions to Codex."""

    model: str
    reasoning_effort: str
    executable: str
    recommended_episodes_per_submission: int | None
    minimum_candidate_evidence: int | None
    max_train_index_uses: int | None

    def __post_init__(self) -> None:
        for field, value in (
            (
                "recommended_episodes_per_submission",
                self.recommended_episodes_per_submission,
            ),
            ("minimum_candidate_evidence", self.minimum_candidate_evidence),
            ("max_train_index_uses", self.max_train_index_uses),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field} must be a positive integer or None")

    def build_invocation(self, task: AgentTask) -> AgentInvocation:
        base = Codex(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            executable=self.executable,
            view_image=True,
        ).build_invocation(task)
        limit = self.max_train_index_uses
        instructions = [_agent_analysis_python_instruction()]
        if (
            self.recommended_episodes_per_submission is not None
            or self.minimum_candidate_evidence is not None
        ):
            instructions.append(
                _training_batch_evidence_instruction(
                    recommended_episodes_per_submission=(
                        self.recommended_episodes_per_submission
                    ),
                    minimum_candidate_evidence=self.minimum_candidate_evidence,
                )
            )
        if limit is not None:
            instructions.append(_training_index_diversity_instruction(limit))
        instruction = "\n".join(instructions)
        option = (
            "-c",
            f"developer_instructions={json.dumps(instruction)}",
        )
        identity = dict(base.identity)
        identity["agent_python_tools"] = "numpy,pillow"
        if self.recommended_episodes_per_submission is not None:
            identity["recommended_episodes_per_submission"] = str(
                self.recommended_episodes_per_submission
            )
        if self.minimum_candidate_evidence is not None:
            identity["minimum_candidate_evidence"] = str(
                self.minimum_candidate_evidence
            )
        if limit is not None:
            identity["max_train_index_uses"] = str(limit)
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


def _agent_analysis_python_instruction() -> str:
    return """\
Crafter feedback analysis uses the launcher's uv-managed Python environment.
Run analysis scripts with `python` directly; that interpreter provides NumPy
and Pillow for loading NPZ observations and producing images. Do not prefix
analysis commands with bare `uv run` from the Run workspace: uv would discover
the repository-root Kernel project instead of the independently packaged
Crafter environment, and that Kernel environment intentionally does not own
Crafter's dependencies. This tooling guidance does not prescribe which
evidence to inspect or how to change the Policy.
"""


def _training_batch_evidence_instruction(
    *,
    recommended_episodes_per_submission: int | None,
    minimum_candidate_evidence: int | None,
) -> str:
    guidance: list[str] = [
        "Crafter training returns have substantial between-Episode variance. "
        "Batches of 8 or fewer Episodes are useful for smoke tests and targeted "
        "diagnosis, but differences between their raw means are weak evidence "
        "about Policy quality."
    ]
    if recommended_episodes_per_submission is not None:
        guidance.append(
            "For ordinary candidate comparisons, normally use about "
            f"{recommended_episodes_per_submission} Episodes per submission."
        )
    if minimum_candidate_evidence is not None:
        guidance.append(
            "Before rejecting a promising Policy direction or deciding that no "
            "further Policy improvement is justified, normally accumulate at "
            f"least {minimum_candidate_evidence} total training Episode results "
            "for that exact submitted Program revision; this evidence may span "
            "multiple submissions."
        )
    guidance.extend(
        [
            "Combine deliberate matched-index comparisons with fresh, unseen "
            "indices for generalization evidence. A matched comparison requires "
            "the exact same submitted Program revision; a hand-written recreation "
            "or approximate revert is a different Policy.",
            "These are statistical evidence guidelines, not Kernel-enforced "
            "submission sizes. You remain responsible for allocating the finite "
            "training budget, may use smaller diagnostic batches when justified, "
            "and may finish early under the Host finish policy.",
        ]
    )
    return (
        "Training batch evidence guidance for this Run. "
        + " ".join(guidance)
        + "\n"
    )


def _training_index_diversity_instruction(max_uses: int) -> str:
    if max_uses == 1:
        reuse_rule = (
            "Each Run-local training Episode index may be selected at most once "
            "across the entire Run. Never select an index that has already been "
            "evaluated."
        )
    else:
        retries = max_uses - 1
        retry_noun = "retry" if retries == 1 else "retries"
        reuse_rule = (
            "Across the entire Run, select each Run-local training Episode index "
            f"at most {max_uses} times in total: its first evaluation plus at most "
            f"{retries} {retry_noun}. A repeat evaluation of an index is permitted only "
            "for a deliberate matched comparison between Policy revisions. Never "
            f"select any index more than {max_uses} times."
        )
    return f"""\
Training Episode diversity is an explicit requirement for this Run. {reuse_rule}
While any unseen training indices remain, every new submission must include
unseen indices and should prefer expanding unique Episode coverage over
repeating previous evidence. Keep a simple index-usage record under analysis/
and check it before every submission. This requirement supplements the Host
statement that index reuse is technically permitted; it limits that permission
in order to preserve rollout diversity. It does not prescribe a particular
selector or submission size, and it does not alter the Host's finish budget
policy.
"""


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if (
        namespace.recommended_episodes_per_submission is not None
        and namespace.recommended_episodes_per_submission
        > namespace.max_episodes_per_submission
    ):
        parser.error(
            "--recommended-episodes-per-submission cannot exceed "
            "--max-episodes-per-submission"
        )
    if (
        namespace.minimum_candidate_evidence is not None
        and namespace.minimum_candidate_evidence > namespace.episode_budget
    ):
        parser.error(
            "--minimum-candidate-evidence cannot exceed --episode-budget"
        )
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
    )
    benchmark_types = {
        "canonical": CrafterBenchmark,
        "canonical-survival": CrafterCanonicalSurvivalBenchmark,
        "canonical-survival-repeat": (
            CrafterCanonicalSurvivalRepeatBenchmark
        ),
        "canonical-strong-survival-repeat": (
            CrafterCanonicalStrongSurvivalRepeatBenchmark
        ),
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
        agent=_CrafterCodex(
            model=namespace.model,
            reasoning_effort=namespace.reasoning_effort,
            executable=namespace.codex_executable,
            recommended_episodes_per_submission=(
                namespace.recommended_episodes_per_submission
            ),
            minimum_candidate_evidence=namespace.minimum_candidate_evidence,
            max_train_index_uses=namespace.max_train_index_uses,
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
                    Path(__file__).parents[1]
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
        choices=(
            "survival-development",
            "long-horizon",
            "canonical",
            "canonical-survival",
            "canonical-survival-repeat",
            "canonical-strong-survival-repeat",
        ),
        default="survival-development",
    )
    parser.add_argument("--max-episode-steps", type=int, default=10_000)
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
        "--recommended-episodes-per-submission",
        type=_recommended_episodes_per_submission,
        default=32,
        metavar="N|none",
        help=(
            "tell Codex to normally use about N Episodes for an ordinary "
            "candidate comparison (default: 32); use 'none' to omit this "
            "launcher-level guidance"
        ),
    )
    parser.add_argument(
        "--minimum-candidate-evidence",
        type=_minimum_candidate_evidence,
        default=64,
        metavar="N|none",
        help=(
            "tell Codex to normally collect at least N total train Episode "
            "results for an exact candidate before rejecting it or finishing "
            "(default: 64); use 'none' to omit this launcher-level guidance"
        ),
    )
    parser.add_argument(
        "--max-train-index-uses",
        type=_max_train_index_uses,
        default=2,
        metavar="N|none",
        help=(
            "tell Codex to select each train Episode index at most N times "
            "across the Run (default: 2); use 'none' to disable this "
            "launcher-level instruction"
        ),
    )
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


def _max_train_index_uses(value: str) -> int | None:
    return _optional_positive_integer(value, label="max train index uses")


def _recommended_episodes_per_submission(value: str) -> int | None:
    return _optional_positive_integer(
        value, label="recommended episodes per submission"
    )


def _minimum_candidate_evidence(value: str) -> int | None:
    return _optional_positive_integer(value, label="minimum candidate evidence")


def _optional_positive_integer(value: str, *, label: str) -> int | None:
    if value.lower() == "none":
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{label} must be a positive integer or 'none'"
        ) from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"{label} must be a positive integer or 'none'"
        )
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
