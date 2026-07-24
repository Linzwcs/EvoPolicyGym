"""Jackdaw CLI — top-level entry point.

Usage::

    jackdaw validate                          # run all scenarios
    jackdaw validate --category jokers        # run only joker scenarios
    jackdaw validate --scenario joker_joker   # run one scenario
    jackdaw validate --host 127.0.0.1 --port 12346
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jackdaw",
        description="Headless Balatro simulator for reinforcement learning research",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- validate ------------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help="Validate engine against live Balatro via scenario-based testing",
    )
    p_validate.add_argument(
        "--category",
        choices=("jokers", "tarots", "planets", "spectrals", "boss_blinds", "modifiers", "tags"),
        default=None,
        help="Run only scenarios in this category",
    )
    p_validate.add_argument(
        "--scenario",
        default=None,
        help="Run a single scenario by name",
    )
    p_validate.add_argument(
        "--host",
        default="127.0.0.1",
        help="Balatrobot host (default: 127.0.0.1)",
    )
    p_validate.add_argument(
        "--port",
        type=int,
        default=12346,
        help="Balatrobot port (default: 12346)",
    )
    p_validate.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between actions for animation watching (default: 0.3)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        from jackdaw.cli.validate import run_validate

        sys.exit(
            run_validate(
                category=args.category,
                scenario=args.scenario,
                host=args.host,
                port=args.port,
                delay=args.delay,
            )
        )


if __name__ == "__main__":
    main()
