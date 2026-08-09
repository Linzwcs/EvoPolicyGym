"""Export one artifact from a selected Program's held-out Assessment rollout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from evopolicygym.benchmark import Benchmark
from evopolicygym.evaluation import EvaluationConfig, evaluate
from evopolicygym.execution import ProcessExecution
from evopolicygym.program import Program

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

    benchmark = _benchmark(
        parser,
        factory_reference=namespace.benchmark_factory,
        config_factory_reference=namespace.benchmark_config_factory,
        config_json=namespace.benchmark_config_json,
    )
    result = evaluate(
        Program.from_directory(namespace.program),
        benchmark,
        execution=ProcessExecution.unsafe(),
        config=EvaluationConfig(
            split=namespace.split,
            episodes=1,
            seed=_assessment_seed(namespace.run_seed),
            episode_timeout_seconds=namespace.episode_timeout_seconds,
        ),
    )
    artifact = next(
        (
            item
            for item in result.feedback.artifacts
            if item.media_type == namespace.media_type
        ),
        None,
    )
    if artifact is None:
        available = sorted(
            {item.media_type for item in result.feedback.artifacts}
        )
        parser.error(
            f"Assessment produced no {namespace.media_type!r} artifact; "
            f"available media types: {available}"
        )

    namespace.output.write_bytes(artifact.read_bytes())
    print(
        json.dumps(
            {
                "artifact": str(namespace.output),
                "episode_index": 0,
                "program_digest": result.program_digest,
                "score": result.feedback.score,
                "source_artifact": artifact.name,
                "source_media_type": artifact.media_type,
                "split": namespace.split,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _benchmark(
    parser: argparse.ArgumentParser,
    *,
    factory_reference: str,
    config_factory_reference: str | None,
    config_json: str,
) -> Benchmark:
    factory = _callable_symbol(parser, factory_reference)
    try:
        config_arguments = json.loads(config_json)
    except json.JSONDecodeError as error:
        parser.error(f"--benchmark-config-json is invalid JSON: {error.msg}")
    if type(config_arguments) is not dict:
        parser.error("--benchmark-config-json must contain a JSON object")

    if config_factory_reference is None:
        if config_arguments:
            parser.error(
                "--benchmark-config-factory is required when config arguments are set"
            )
        benchmark = factory()
    else:
        config_factory = _callable_symbol(parser, config_factory_reference)
        config = config_factory(**config_arguments)
        benchmark = factory(config)
    if not isinstance(benchmark, Benchmark):
        parser.error("--benchmark-factory did not return a Benchmark")
    return benchmark


def _callable_symbol(
    parser: argparse.ArgumentParser,
    reference: str,
) -> Callable[..., object]:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        parser.error("symbol references must use MODULE:ATTRIBUTE")
    try:
        module = importlib.import_module(module_name)
        symbol = getattr(module, attribute_name)
    except (ImportError, AttributeError) as error:
        parser.error(f"cannot load {reference!r}: {error}")
    if not callable(symbol):
        parser.error(f"{reference!r} is not callable")
    return cast(Callable[..., object], symbol)


def _assessment_seed(run_seed: int) -> int:
    digest = hashlib.sha256()
    digest.update(_ASSESSMENT_SEED_DOMAIN)
    digest.update(run_seed.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a selected Program on Assessment Episode 0 and export "
            "one exact feedback artifact."
        ),
    )
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-factory", required=True)
    parser.add_argument("--benchmark-config-factory")
    parser.add_argument("--benchmark-config-json", default="{}")
    parser.add_argument("--media-type", default="image/gif")
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--episode-timeout-seconds", type=float, default=1_800)
    parser.add_argument("--allow-unsafe-process", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
