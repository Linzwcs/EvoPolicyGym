"""Run a paired Codex experiment with and without the Balatro skill."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).resolve().parents[1]
_SINGLE_RUNNER = _REPOSITORY / "scripts" / "run_balatro_codex.py"
_BALATRO_SKILL = _REPOSITORY / "skills" / "optimize-balatro-policy"
_RESULT_SCHEMA = "evopolicygym/balatro-skill-ab/v4"


class ExperimentError(RuntimeError):
    """The paired experiment could not produce a comparable result."""


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if not namespace.allow_unsafe_process:
        parser.error(
            "local Agent and Policy processes are not isolated; "
            "pass --allow-unsafe-process to acknowledge this"
        )

    record_root = (
        _default_record_root(namespace.model, namespace.seed)
        if namespace.record_root is None
        else namespace.record_root.resolve()
    )
    if record_root.exists() or record_root.is_symlink():
        parser.error("--record-root must not already exist")
    record_root.parent.mkdir(parents=True, exist_ok=True)
    record_root.mkdir(mode=0o700)

    arms = (
        (("no-skill", False), ("with-skill", True))
        if namespace.order == "no-skill-first"
        else (("with-skill", True), ("no-skill", False))
    )
    results: dict[str, dict[str, Any]] = {}
    try:
        with _short_record_alias(record_root) as execution_root:
            for arm_name, use_skill in arms:
                results[arm_name] = _run_arm(
                    namespace,
                    record_root=record_root,
                    execution_root=execution_root,
                    arm_name=arm_name,
                    use_skill=use_skill,
                )
    except ExperimentError as error:
        print(f"Balatro skill A/B failed: {error}", file=sys.stderr)
        print(f"Partial records: {record_root}", file=sys.stderr)
        return 1

    no_skill = results["no-skill"]
    with_skill = results["with-skill"]
    comparison = {
        "schema": _RESULT_SCHEMA,
        "record_root": str(record_root),
        "model": namespace.model,
        "reasoning_effort": namespace.reasoning_effort,
        "seed": namespace.seed,
        "order": namespace.order,
        "config": {
            "max_submissions": namespace.max_submissions,
            "episode_budget": namespace.episode_budget,
            "episode_pool_size": namespace.episode_pool_size,
            "max_episodes_per_submission": (
                namespace.max_episodes_per_submission
            ),
            "validation_episodes_per_candidate": (
                namespace.validation_episodes_per_candidate
            ),
            "validation_max_candidates": (
                namespace.validation_max_candidates
            ),
            "assessment_episodes": namespace.assessment_episodes,
        },
        "no_skill": no_skill,
        "with_skill": with_skill,
        "assessment_score_delta": (
            with_skill["assessment_score"] - no_skill["assessment_score"]
        ),
    }
    comparison_path = record_root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def _run_arm(
    namespace: argparse.Namespace,
    *,
    record_root: Path,
    execution_root: Path,
    arm_name: str,
    use_skill: bool,
) -> dict[str, Any]:
    record_to = record_root / arm_name
    execution_record_to = execution_root / arm_name
    command = [
        sys.executable,
        str(_SINGLE_RUNNER),
        "--model",
        namespace.model,
        "--codex-executable",
        namespace.codex_executable,
        "--record-to",
        str(execution_record_to),
        "--split",
        namespace.split,
        "--seed",
        str(namespace.seed),
        "--max-submissions",
        str(namespace.max_submissions),
        "--episode-budget",
        str(namespace.episode_budget),
        "--validation-episodes-per-candidate",
        str(namespace.validation_episodes_per_candidate),
        "--validation-split",
        namespace.validation_split,
        "--validation-max-candidates",
        str(namespace.validation_max_candidates),
        "--assessment-episodes",
        str(namespace.assessment_episodes),
        "--assessment-split",
        namespace.assessment_split,
        "--episode-timeout-seconds",
        str(namespace.episode_timeout_seconds),
        "--agent-timeout-seconds",
        str(namespace.agent_timeout_seconds),
        "--progress",
        namespace.progress,
        "--allow-unsafe-process",
        "--reasoning-effort",
        namespace.reasoning_effort,
    ]
    if namespace.episode_pool_size is not None:
        command.extend(
            (
                "--episode-pool-size",
                str(namespace.episode_pool_size),
            )
        )
    command.extend(
        (
            "--max-episodes-per-submission",
            str(namespace.max_episodes_per_submission),
        )
    )
    if use_skill:
        command.extend(("--skill", str(_BALATRO_SKILL)))

    print(
        f"\n=== Balatro skill A/B: {arm_name} ===\n"
        f"Record: {record_to}",
        file=sys.stderr,
        flush=True,
    )
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )
    raw_result_path = record_root / f"{arm_name}-result.json"
    raw_result_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise ExperimentError(
            f"{arm_name} exited with status {completed.returncode}; "
            f"see {record_to}"
        )

    try:
        run_result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ExperimentError(
            f"{arm_name} did not emit a valid JSON result"
        ) from error
    if not isinstance(run_result, dict):
        raise ExperimentError(f"{arm_name} emitted a non-object result")

    assessment = _read_report(
        record_to / "assessment" / "report.json",
        arm_name=arm_name,
        report_name="Assessment",
    )
    validation = _read_report(
        record_to / "validation" / "report.json",
        arm_name=arm_name,
        report_name="Validation",
    )
    assessment_score = assessment.get("score")
    policy_failures = assessment.get("policy_failures")
    if not isinstance(assessment_score, int | float) or isinstance(
        assessment_score, bool
    ):
        raise ExperimentError(f"{arm_name} Assessment score is invalid")
    if not isinstance(policy_failures, int) or isinstance(
        policy_failures, bool
    ):
        raise ExperimentError(
            f"{arm_name} Assessment policy_failures is invalid"
        )

    return {
        "skills": (
            [str(_BALATRO_SKILL.relative_to(_REPOSITORY))]
            if use_skill
            else []
        ),
        "terminal_reason": run_result.get("terminal_reason"),
        "record": str(record_to),
        "final_submission_id": run_result.get("final_submission_id"),
        "validation_selected_submission_id": validation.get(
            "selected_submission_id"
        ),
        "assessment_score": float(assessment_score),
        "assessment_policy_failures": policy_failures,
    }


@contextmanager
def _short_record_alias(record_root: Path) -> Iterator[Path]:
    """Keep AF_UNIX addresses short while retaining records under ``runs/``."""

    short_parent = Path("/tmp")
    if not short_parent.is_dir():
        short_parent = Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix="epg-", dir=short_parent) as temporary:
        alias = Path(temporary) / "r"
        alias.symlink_to(record_root, target_is_directory=True)
        yield alias


def _read_report(
    path: Path,
    *,
    arm_name: str,
    report_name: str,
) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ExperimentError(
            f"{arm_name} produced no {report_name} report; "
            f"inspect {path.parent.parent / 'run.json'}"
        ) from error
    except json.JSONDecodeError as error:
        raise ExperimentError(
            f"{arm_name} {report_name} report is invalid JSON"
        ) from error
    if not isinstance(document, dict):
        raise ExperimentError(
            f"{arm_name} {report_name} report is not an object"
        )
    return document


def _default_record_root(model: str, seed: int) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
    if not model_slug:
        model_slug = "model"
    return (
        _REPOSITORY
        / "runs"
        / f"balatro-skill-ab-{model_slug}-seed{seed}-{timestamp}"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _unsigned_int(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**64 - 1:
        raise argparse.ArgumentTypeError(
            "must be an unsigned 64-bit integer"
        )
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched Balatro Codex experiments with and without the "
            "selected Balatro Agent Skill, then compare held-out Assessment "
            "scores."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort",
        required=True,
        help=(
            "use the same Codex reasoning effort for both experiment arms"
        ),
    )
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument(
        "--record-root",
        type=Path,
        help="new parent directory for both Run records",
    )
    parser.add_argument(
        "--order",
        choices=("no-skill-first", "with-skill-first"),
        default="no-skill-first",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--seed", type=_unsigned_int, default=0)
    parser.add_argument(
        "--max-submissions",
        type=_positive_int,
        default=16,
    )
    parser.add_argument(
        "--episode-budget",
        type=_positive_int,
        default=48,
    )
    parser.add_argument(
        "--episode-pool-size",
        type=_positive_int,
        help="fixed selectable pool size; defaults to the Episode budget",
    )
    parser.add_argument(
        "--max-episodes-per-submission",
        type=_positive_int,
        default=3,
    )
    parser.add_argument(
        "--validation-episodes-per-candidate",
        type=_positive_int,
        default=8,
    )
    parser.add_argument(
        "--validation-split",
        choices=("train", "validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--validation-max-candidates",
        type=_positive_int,
        default=3,
    )
    parser.add_argument(
        "--assessment-episodes",
        type=_positive_int,
        default=32,
    )
    parser.add_argument(
        "--assessment-split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--episode-timeout-seconds",
        type=_positive_float,
        default=60.0,
    )
    parser.add_argument(
        "--agent-timeout-seconds",
        type=_positive_float,
        default=7_200.0,
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="plain",
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
