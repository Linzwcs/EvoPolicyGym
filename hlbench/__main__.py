"""Top-level HLBench command dispatcher."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m hlbench {scenario|rollout|run|seeds} ...")
        return 0
    command = args.pop(0)
    if command == "scenario":
        from hlbench.core.scenario_cli import main as scenario_main

        return scenario_main(args)
    if command == "rollout":
        from hlbench.rollout.cli import main as rollout_main

        return rollout_main(args)
    if command == "run":
        from hlbench.harness.cli import main as harness_main

        return harness_main(args)
    if command == "seeds":
        from hlbench.core.seed_cli import main as seed_main

        return seed_main(args)
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
