"""Policy evaluator used by the harness."""

from __future__ import annotations

import py_compile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.core.artifacts import write_json, write_jsonl
from hlbench.core.scenario import ScenarioSplit, load_scenario
from hlbench.rollout.engine import RolloutResult, purge_private_artifacts, run_rollout


@dataclass(frozen=True)
class CompileResult:
    ok: bool
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {"ok": self.ok, "error": self.error}


def compile_policy(policy_path: Path) -> CompileResult:
    try:
        py_compile.compile(str(policy_path), doraise=True)
    except py_compile.PyCompileError as exc:
        return CompileResult(ok=False, error=str(exc))
    return CompileResult(ok=True)


def evaluate_split(
    *,
    scenario_name: str,
    split: str,
    policy_path: Path,
    output_dir: Path,
    episodes: int | None = None,
    sampler_seed: int | None = None,
) -> RolloutResult:
    try:
        return run_rollout(
            scenario_name=scenario_name,
            split=split,
            policy_path=policy_path,
            episodes=episodes,
            sampler_seed=sampler_seed,
            output_dir=output_dir,
            write_episode_artifacts=(split == "train"),
        )
    except Exception as exc:
        return minimum_score_rollout(
            scenario_name=scenario_name,
            split=split,
            output_dir=output_dir,
            episodes=episodes,
            error=repr(exc),
        )


def minimum_score_rollout(
    *,
    scenario_name: str,
    split: str,
    output_dir: Path,
    episodes: int | None,
    error: str,
) -> RolloutResult:
    scenario = load_scenario(scenario_name)
    split = ScenarioSplit(split).value
    private_split = split != ScenarioSplit.TRAIN.value
    seeds = scenario.seeds_for_split(split, limit=episodes)
    output_dir.mkdir(parents=True, exist_ok=True)
    if private_split:
        purge_private_artifacts(output_dir)
    summary = {
        "episodes": len(seeds),
        "successes": 0,
        "success_rate": 0.0,
        "mean_score": scenario.minimum_score.value,
        "mean_steps": 0.0,
        "invalid_actions": 0,
        "minimum_score_episodes": len(seeds),
    }
    failures = [] if private_split else [{"terminated_by": "evaluator_exception", "exception": error}]
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "manifest.json",
        {
            "scenario": scenario.name,
            "scenario_id": scenario.scenario_id,
            "split": split,
            "private_split": private_split,
            "seed_pool_count": len(scenario.splits[split].seeds),
            "sampled_episodes": len(seeds),
            "minimum_score_applied": True,
        },
    )
    if not private_split:
        write_jsonl(output_dir / "episodes.jsonl", [])
        write_jsonl(output_dir / "failures.jsonl", failures)
    return RolloutResult(
        scenario=scenario.name,
        split=split,
        run_dir=output_dir,
        trials=[],
        summary=summary,
        failures=failures,
    )
