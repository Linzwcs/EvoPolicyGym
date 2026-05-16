"""Seed pool command line tools."""

from __future__ import annotations

import argparse
import json

from hlbench.core.seeds import DEFAULT_MAX_SEED, SeedGenerationConfig, write_seed_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="generate fixed random split seed files")
    generate.add_argument("--pool", default="default")
    generate.add_argument("--generator-seed", type=int, required=True)
    generate.add_argument("--train-count", type=int, default=10_000)
    generate.add_argument("--validation-count", type=int, default=200)
    generate.add_argument("--heldout-count", type=int, default=200)
    generate.add_argument("--min-seed", type=int, default=0)
    generate.add_argument("--max-seed", type=int, default=DEFAULT_MAX_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        config = SeedGenerationConfig(
            generator_seed=args.generator_seed,
            train_count=args.train_count,
            validation_count=args.validation_count,
            heldout_count=args.heldout_count,
            min_seed=args.min_seed,
            max_seed=args.max_seed,
        )
        written = write_seed_files(pool_name=args.pool, config=config)
        print(json.dumps({split: str(path) for split, path in written.items()}, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
