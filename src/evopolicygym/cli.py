"""Operator-facing command-line presentation over the public SDK."""

from __future__ import annotations

import argparse

from ._version import __version__


def main(arguments: list[str] | None = None) -> int:
    parser = _parser()
    parser.parse_args(arguments)
    parser.print_help()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evopolicygym",
        description=(
            "Operator commands for EvoPolicyGym. Evaluation and Run workflows "
            "are currently available through the public Python SDK."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
