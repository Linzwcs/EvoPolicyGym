"""Generic harness CLI."""

from __future__ import annotations

import argparse
import json
import shlex

from hlbench.harness.agents.config import AGENT_PRESETS
from hlbench.harness.loop_runner import run_loop
from hlbench.harness.presets import get_preset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-name", default="local")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--agent-backend", choices=("command",), default="command")
    parser.add_argument("--agent-preset", choices=sorted(AGENT_PRESETS), default="none")
    parser.add_argument("--agent-command", default=None)
    parser.add_argument("--command", default=None, help="Deprecated alias for --agent-command.")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command and args.agent_command:
        raise SystemExit("use only one of --command or --agent-command")
    preset = get_preset(args.preset)
    agent_command = args.agent_command if args.agent_command is not None else args.command
    result = run_loop(
        scenario_name=args.scenario,
        epochs=args.epochs,
        run_id=args.run_id,
        agent_backend=args.agent_backend,
        agent_preset=args.agent_preset,
        agent_command=shlex.split(agent_command) if agent_command else None,
        model_name=args.model_name,
        train_episodes=args.train_episodes if args.train_episodes is not None else preset.train_episodes,
        timeout_seconds=args.timeout_seconds if args.timeout_seconds is not None else preset.timeout_seconds,
    )
    print(json.dumps(result.to_record(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
