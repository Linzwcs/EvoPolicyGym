"""Build learner-visible context for a scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = REPO_ROOT / "hlbench" / "scenarios"


def build_context(scenario_name: str, run_dir: Path | None = None) -> dict[str, Any]:
    scenario_dir = SCENARIO_ROOT / scenario_name
    scenario = json.loads((scenario_dir / "scenario.json").read_text())
    learner_scenario = {
        "scenario_id": scenario["scenario_id"],
        "env_id": scenario["env_id"],
        "train_seeds": scenario["train_seeds"],
        "validation_seed_count": len(scenario["validation_seeds"]),
        "heldout_seed_count": len(scenario["heldout_seeds"]),
        "max_steps": scenario["max_steps"],
        "allowed_files": scenario["allowed_files"],
        "protected_files": scenario["protected_files"],
    }

    context: dict[str, Any] = {
        "scenario": learner_scenario,
        "task_spec": (scenario_dir / "task_spec.md").read_text(),
        "policy_py": (scenario_dir / "policy.py").read_text(),
        "rollout_summary": None,
        "failure_modes": [],
    }
    if run_dir is not None:
        summary_path = run_dir / "rollout" / "summary.json"
        failures_path = run_dir / "rollout" / "failures.jsonl"
        if summary_path.exists():
            context["rollout_summary"] = json.loads(summary_path.read_text())
        if failures_path.exists():
            context["failure_modes"] = [
                json.loads(line) for line in failures_path.read_text().splitlines() if line.strip()
            ]
    return context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="minigrid_doorkey")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    context = build_context(args.scenario, args.run_dir)
    text = json.dumps(context, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

