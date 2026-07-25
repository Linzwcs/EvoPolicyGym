"""Export one deterministic baseline replay for the documentation site."""

from __future__ import annotations

import argparse
from pathlib import Path

from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.execution import ProcessExecution

from balatro import BalatroBenchmark, baseline_program


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    result = evaluate(
        baseline_program(),
        BalatroBenchmark(),
        execution=ProcessExecution.unsafe(),
        config=EvaluationConfig(
            split="validation",
            episodes=1,
            seed=args.seed,
            episode_timeout_seconds=30,
        ),
    )
    replay = next(
        artifact for artifact in result.feedback.artifacts if artifact.name == "replay.jsonl"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(replay.read_bytes())
    print(
        f"wrote {args.output} ({result.episodes[0].steps} steps, score {result.feedback.score:.0f})"
    )


if __name__ == "__main__":
    main()
