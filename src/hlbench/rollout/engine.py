"""Rollout engine."""

from __future__ import annotations

import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.core.artifacts import write_json, write_jsonl
from hlbench.core.paths import run_root
from hlbench.core.policy import file_sha256, load_policy, reset_policy
from hlbench.core.scenario import ScenarioSplit, ScenarioSpec, load_scenario
from hlbench.envs.base import EnvironmentInstance
from hlbench.envs.registry import get_backend
from hlbench.rollout.summarize import failure_samples, summarize_trials

PRIVATE_SPLITS = {ScenarioSplit.VALIDATION.value, ScenarioSplit.HELDOUT.value}


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    seed: int
    score: float
    steps: int
    terminated: bool
    truncated: bool
    terminated_by: str
    invalid_actions: int = 0
    exception: str | None = None
    success: bool = False
    minimum_score_applied: bool = False

    def to_record(self, *, include_seed: bool) -> dict[str, Any]:
        record = {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "score": self.score,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "terminated_by": self.terminated_by,
            "invalid_actions": self.invalid_actions,
            "exception": self.exception,
            "success": self.success,
            "minimum_score_applied": self.minimum_score_applied,
        }
        if not include_seed:
            record.pop("seed")
        return record


@dataclass(frozen=True)
class RolloutResult:
    scenario: str
    split: str
    run_dir: Path
    trials: list[dict[str, Any]]
    summary: dict[str, Any]
    failures: list[dict[str, Any]]

    def to_record(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "split": self.split,
            "run_dir": str(self.run_dir),
            "trials": self.trials,
            "summary": self.summary,
            "failures": self.failures,
        }


def default_run_dir(
    *,
    scenario: ScenarioSpec,
    model_name: str,
    run_id: str | None = None,
) -> Path:
    run_name = run_id or time.strftime("%Y%m%d-%H%M%S")
    return run_root(model_name=model_name, env_name=scenario.name, run_id=run_name)


def make_backend(scenario: ScenarioSpec) -> EnvironmentInstance:
    return get_backend(scenario.env_backend).make(scenario)


def run_episode(
    *,
    backend: EnvironmentInstance,
    policy: Any,
    scenario: ScenarioSpec,
    seed: int,
    task_config: dict[str, Any] | None = None,
    replay_path: Path | None = None,
) -> EpisodeResult:
    replay: list[dict[str, Any]] = []
    try:
        observation = backend.reset(seed=seed, config=task_config)
        reset_policy(policy, task_config=task_config or {})
    except Exception as exc:
        return EpisodeResult(
            episode_id=task_config.get("episode_id", "episode_unknown") if task_config else "episode_unknown",
            seed=seed,
            score=scenario.minimum_score.value,
            steps=0,
            terminated=False,
            truncated=False,
            terminated_by="reset_exception",
            exception=repr(exc),
            minimum_score_applied=True,
        )

    total_reward = 0.0
    invalid_actions = 0
    terminated = False
    truncated = False
    steps = 0
    exception: str | None = None

    while steps < scenario.max_steps:
        before = observation
        try:
            action = policy.act(
                observation,
                {
                    "action_schema": backend.action_schema,
                    "action_count": backend.action_count,
                    "step_index": steps,
                    "max_steps": scenario.max_steps,
                },
            )
        except Exception as exc:
            exception = repr(exc)
            terminated_by = "policy_exception"
            break

        try:
            step = backend.step(action)
        except Exception as exc:
            invalid_actions += 1
            exception = repr(exc)
            replay.append({"t": steps, "observation": before, "action": repr(action), "error": exception})
            terminated_by = "invalid_action"
            break

        replay.append(
            {
                "t": steps,
                "observation": before,
                "action": action,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "info": step.info,
            }
        )
        total_reward += step.reward
        observation = step.observation
        terminated = step.terminated
        truncated = step.truncated
        steps += 1
        if step.done:
            terminated_by = "terminated" if step.terminated else "truncated"
            break
    else:
        terminated_by = "timeout"
        truncated = True

    if replay_path is not None:
        write_jsonl(replay_path, replay)
    minimum_score_applied = exception is not None
    success_threshold = float(scenario.metadata.get("success_score", scenario.max_steps))
    score = scenario.minimum_score.value if minimum_score_applied else total_reward
    success = not minimum_score_applied and score >= success_threshold
    return EpisodeResult(
        episode_id=task_config.get("episode_id", "episode_unknown") if task_config else "episode_unknown",
        seed=seed,
        score=score,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
        terminated_by=terminated_by,
        invalid_actions=invalid_actions,
        exception=exception,
        success=success,
        minimum_score_applied=minimum_score_applied,
    )


def run_rollout(
    *,
    scenario_name: str,
    split: str,
    policy_path: Path,
    episodes: int | None = None,
    run_id: str | None = None,
    model_name: str = "local",
    sampler_seed: int | None = None,
    output_dir: Path | None = None,
    write_episode_artifacts: bool | None = None,
) -> RolloutResult:
    scenario = load_scenario(scenario_name)
    split = ScenarioSplit(split).value
    private_split = split in PRIVATE_SPLITS
    effective_sampler_seed = _effective_sampler_seed(episodes=episodes, sampler_seed=sampler_seed)
    seeds = scenario.seeds_for_split(split, limit=episodes, sampler_seed=effective_sampler_seed)
    run_dir = output_dir or default_run_dir(scenario=scenario, model_name=model_name, run_id=run_id) / split
    run_dir.mkdir(parents=True, exist_ok=True)
    if private_split:
        purge_private_artifacts(run_dir)
    write_artifacts = (split == ScenarioSplit.TRAIN.value) if write_episode_artifacts is None else write_episode_artifacts
    write_artifacts = bool(write_artifacts and not private_split)

    try:
        policy = load_policy(policy_path)
        backend = make_backend(scenario)
    except Exception as exc:
        return _minimum_rollout_result(
            scenario=scenario,
            split=split,
            run_dir=run_dir,
            policy_path=policy_path,
            seeds=seeds,
            private_split=private_split,
            error=repr(exc),
        )
    internal_trials: list[dict[str, Any]] = []
    public_trials: list[dict[str, Any]] = []
    try:
        for index, seed in enumerate(seeds):
            episode_id = f"{split}_{index:06d}"
            replay_path = run_dir / "replays" / f"episode_{index:04d}.jsonl" if write_artifacts else None
            result = run_episode(
                backend=backend,
                policy=policy,
                scenario=scenario,
                seed=seed,
                task_config={"scenario": scenario.scenario_id, "episode_id": episode_id},
                replay_path=replay_path,
            )
            record = result.to_record(include_seed=True)
            public_record = result.to_record(include_seed=False)
            if replay_path is not None:
                record["replay_path"] = str(replay_path)
                public_record["replay_path"] = str(replay_path)
            internal_trials.append(record)
            if not private_split:
                public_trials.append(public_record)
    finally:
        backend.close()

    summary = summarize_trials(internal_trials)
    failures = [] if private_split else failure_samples(public_trials)
    write_json(run_dir / "summary.json", summary)
    if not private_split:
        write_jsonl(run_dir / "episodes.jsonl", public_trials)
        write_jsonl(run_dir / "failures.jsonl", failures)
    write_json(
        run_dir / "manifest.json",
        {
            "scenario": scenario.name,
            "scenario_id": scenario.scenario_id,
            "env_id": scenario.env_id,
            "split": split,
            "policy_path": str(policy_path),
            "policy_sha256": file_sha256(policy_path),
            "episode_artifacts": bool(write_artifacts),
            "private_split": private_split,
            "seed_pool_count": len(scenario.splits[split].seeds),
            "sampled_episodes": len(seeds),
            "sampler_seed": effective_sampler_seed,
        },
    )
    return RolloutResult(
        scenario=scenario.name,
        split=split,
        run_dir=run_dir,
        trials=public_trials,
        summary=summary,
        failures=failures,
    )


def _minimum_rollout_result(
    *,
    scenario: ScenarioSpec,
    split: str,
    run_dir: Path,
    policy_path: Path,
    seeds: list[int],
    private_split: bool,
    error: str,
) -> RolloutResult:
    summary = {
        "episodes": len(seeds),
        "successes": 0,
        "success_rate": 0.0,
        "mean_score": scenario.minimum_score.value,
        "mean_steps": 0.0,
        "invalid_actions": 0,
        "minimum_score_episodes": len(seeds),
    }
    failures = [] if private_split else [{"terminated_by": "setup_exception", "exception": error}]
    write_json(run_dir / "summary.json", summary)
    if not private_split:
        write_jsonl(run_dir / "episodes.jsonl", [])
        write_jsonl(run_dir / "failures.jsonl", failures)
    policy_sha = file_sha256(policy_path) if policy_path.exists() else None
    write_json(
        run_dir / "manifest.json",
        {
            "scenario": scenario.name,
            "scenario_id": scenario.scenario_id,
            "env_id": scenario.env_id,
            "split": split,
            "policy_path": str(policy_path),
            "policy_sha256": policy_sha,
            "episode_artifacts": False,
            "private_split": private_split,
            "sampled_episodes": len(seeds),
            "minimum_score_applied": True,
        },
    )
    return RolloutResult(
        scenario=scenario.name,
        split=split,
        run_dir=run_dir,
        trials=[],
        summary=summary,
        failures=failures,
    )


def purge_private_artifacts(run_dir: Path) -> None:
    """Remove public-trace artifacts before writing a private split result."""
    for name in ("episodes.jsonl", "failures.jsonl", "trials.json"):
        path = run_dir / name
        if path.exists():
            path.unlink()
    for name in ("replays", "episodes"):
        path = run_dir / name
        if path.exists():
            shutil.rmtree(path)


def _effective_sampler_seed(*, episodes: int | None, sampler_seed: int | None) -> int | None:
    if episodes is None:
        return sampler_seed
    if sampler_seed is not None:
        return sampler_seed
    return time.time_ns()
