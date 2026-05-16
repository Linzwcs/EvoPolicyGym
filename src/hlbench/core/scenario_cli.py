"""Scenario command line tools."""

from __future__ import annotations

import argparse
import json

from hlbench.core.validate import validate_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a scenario contract and environment smoke step")
    validate.add_argument("--scenario", required=True)
    validate.add_argument("--no-smoke-step", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_scenario(args.scenario, smoke_step=not args.no_smoke_step)
        print(json.dumps(result.to_record(), indent=2, sort_keys=True))
        return 0 if result.ok else 1
    raise AssertionError(args.command)
