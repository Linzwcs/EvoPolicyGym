"""Export one held-out Assessment rollout from an immutable Program."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evopolicygym.evaluation import EvaluationConfig, evaluate
from evopolicygym.execution import ProcessExecution
from evopolicygym.program import Program

from robotics_benchmarks import ROBOTICS_PROFILES, RoboticsBenchmark, RoboticsConfig

_ASSESSMENT_SEED_DOMAIN = b"evopolicygym/assessment-seed/v1\0"


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    namespace = parser.parse_args(arguments)
    if not namespace.allow_unsafe_process:
        parser.error(
            "the Policy process is not isolated; pass --allow-unsafe-process "
            "to acknowledge this"
        )
    if namespace.output.exists() or namespace.output.is_symlink():
        parser.error("--output must not already exist")
    if not namespace.output.parent.is_dir():
        parser.error("--output parent directory must exist")

    result = evaluate(
        Program.from_directory(namespace.program),
        RoboticsBenchmark(RoboticsConfig(profile=namespace.profile)),
        execution=ProcessExecution.unsafe(),
        config=EvaluationConfig(
            split="test",
            episodes=1,
            seed=_assessment_seed(namespace.run_seed),
            episode_timeout_seconds=namespace.episode_timeout_seconds,
        ),
    )
    preview = next(
        artifact
        for artifact in result.feedback.artifacts
        if artifact.media_type == "image/gif"
    )
    namespace.output.write_bytes(preview.read_bytes())
    print(
        json.dumps(
            {
                "artifact": str(namespace.output),
                "episode_index": 0,
                "program_digest": result.program_digest,
                "score": result.feedback.score,
                "split": "test",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _assessment_seed(run_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(_ASSESSMENT_SEED_DOMAIN)
    digest.update(run_seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the first held-out Robotics Assessment rollout as GIF.",
    )
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=ROBOTICS_PROFILES, required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--episode-timeout-seconds", type=float, default=1_800)
    parser.add_argument("--allow-unsafe-process", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
