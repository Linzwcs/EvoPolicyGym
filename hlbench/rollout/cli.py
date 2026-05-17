"""Rollout command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hlbench.rollout.engine import run_rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--split", choices=("train", "validation", "heldout"), default="train")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-name", default="local")
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-episode-artifacts", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workspace is not None and args.split != "train":
        raise SystemExit("workspace rollouts may only use --split train")
    scenario = args.scenario or _scenario_from_workspace(args.workspace)
    if scenario is None:
        raise SystemExit("--scenario is required unless --workspace contains task_contract.json")
    policy = args.policy
    if policy is None and args.workspace is not None:
        policy = args.workspace / "system" / "policy.py"
    if policy is None:
        from hlbench.core.scenario import find_scenario_dir

        policy = find_scenario_dir(scenario) / "policy.py"
    result = run_rollout(
        scenario_name=scenario,
        split=args.split,
        policy_path=policy,
        episodes=args.episodes,
        run_id=args.run_id,
        model_name=args.model_name,
        sampler_seed=args.sampler_seed,
        output_dir=args.output_dir,
        write_episode_artifacts=True if args.write_episode_artifacts else None,
    )
    print(json.dumps(result.to_record(), indent=2, sort_keys=True))
    return 0


def _scenario_from_workspace(workspace: Path | None) -> str | None:
    if workspace is None:
        return None
    contract_path = workspace / "task_contract.json"
    if not contract_path.exists():
        return None
    data: dict[str, Any] = json.loads(contract_path.read_text())
    scenario = data.get("scenario", {})
    name = scenario.get("name")
    return str(name) if name else None


if __name__ == "__main__":
    raise SystemExit(main())
