"""Terminal immutable run.json manifest projection."""

from __future__ import annotations

from pathlib import Path

from ..._version import __version__
from ...benchmark import BenchmarkSpec
from ...program import Program
from ...results import RunResult
from ...skills import AgentSkill
from .. import RunConfig
from .._agent import AgentOutcome
from .._episode_pool import (
    TRAINING_POLICY_DERIVATION,
    TRAINING_POOL_DERIVATION,
)
from .._json import encode_public_json_value
from .writer import write_json_atomic

_RUN_RECORD_SCHEMA = "evopolicygym/run-record/v6"


def write_run_manifest(
    path: Path,
    result: RunResult,
    *,
    benchmark_spec: BenchmarkSpec,
    initial_program: Program,
    config: RunConfig,
    skills: tuple[AgentSkill, ...],
    agent_outcome: AgentOutcome,
    agent_identity: dict[str, str],
) -> None:
    submissions = [
        {
            "submission_id": item.submission_id,
            "program_digest": item.program_digest,
            "episode_indices": list(item.episode_indices),
            "episodes_used": item.episodes_used,
            "episodes_remaining": item.episodes_remaining,
            "score": item.feedback.score,
            "record": f"submissions/{item.submission_id}",
        }
        for item in result.submissions
    ]
    write_json_atomic(
        path,
        {
            "schema": _RUN_RECORD_SCHEMA,
            "library_version": __version__,
            "benchmark": {
                "id": benchmark_spec.id,
                "environment": {
                    "digest": benchmark_spec.environment_digest,
                    "parameters": encode_public_json_value(
                        benchmark_spec.environment_parameters
                    ),
                },
            },
            "initial_program": {
                "digest": initial_program.digest,
                "record": "initial/program",
            },
            "workspace": {
                "root": "workspace",
                "program": "workspace/program",
                "feedback": "workspace/feedback",
                **(
                    {"skills": "workspace/skills"}
                    if skills
                    else {}
                ),
            },
            "skills": [
                {
                    "name": skill.name,
                    "digest": skill.digest,
                    "record": f"workspace/skills/{skill.name}",
                }
                for skill in skills
            ],
            "events": "events.jsonl",
            "config": {
                "split": config.split,
                "seed": config.seed,
                "max_submissions": config.max_submissions,
                "episode_budget": config.episode_budget,
                "episode_pool_size": config.episode_pool_size,
                "max_episodes_per_submission": (
                    config.max_episodes_per_submission
                ),
                "episode_timeout_seconds": config.episode_timeout_seconds,
                "agent_timeout_seconds": config.agent_timeout_seconds,
                "validation": (
                    None
                    if config.validation is None
                    else {
                        "split": config.validation.split,
                        "episodes_per_candidate": (
                            config.validation.episodes_per_candidate
                        ),
                        "max_candidates": config.validation.max_candidates,
                    }
                ),
                "assessment": (
                    None
                    if config.assessment is None
                    else {
                        "split": config.assessment.split,
                        "episodes": config.assessment.episodes,
                    }
                ),
            },
            "training_episode_pool": {
                "size": config.episode_pool_size,
                "episode_derivation": TRAINING_POOL_DERIVATION,
                "policy_seed_derivation": TRAINING_POLICY_DERIVATION,
            },
            "agent": {
                **agent_identity,
                "invocation": "agent/invocation.json",
                "stdout": "agent/stdout.log",
                "stderr": "agent/stderr.log",
                "timed_out": agent_outcome.timed_out,
                "stopped_after_terminal": (
                    agent_outcome.stopped_after_terminal
                ),
                "start_failed": agent_outcome.start_failed,
                "returncode": agent_outcome.returncode,
            },
            "terminal_reason": result.terminal_reason,
            "final_submission_id": result.final_submission_id,
            "candidate_submission_ids": list(
                result.candidate_submission_ids
            ),
            "validation": (
                None
                if result.validation is None
                else {"report": "validation/report.json"}
            ),
            "assessment": (
                None
                if result.assessment is None
                else {"report": "assessment/report.json"}
            ),
            "submissions": submissions,
        },
    )


__all__: list[str] = []
