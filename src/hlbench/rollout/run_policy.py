"""Run a scenario policy and emit rollout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hlbench.adapters.gymnasium_minigrid import GymnasiumMinigridAdapter
from hlbench.rollout.summarize import summarize_trials


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_ROOT = REPO_ROOT / "hlbench" / "scenarios"
RUNS_ROOT = REPO_ROOT / "runs"

ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    env_id: str
    max_steps: int
    train_seeds: list[int]
    validation_seeds: list[int]
    heldout_seeds: list[int]

    @classmethod
    def load(cls, scenario_name: str) -> "Scenario":
        path = SCENARIO_ROOT / scenario_name / "scenario.json"
        data = json.loads(path.read_text())
        return cls(
            scenario_id=str(data["scenario_id"]),
            env_id=str(data["env_id"]),
            max_steps=int(data["max_steps"]),
            train_seeds=[int(seed) for seed in data["train_seeds"]],
            validation_seeds=[int(seed) for seed in data["validation_seeds"]],
            heldout_seeds=[int(seed) for seed in data["heldout_seeds"]],
        )

    def seeds_for_split(self, split: str) -> list[int]:
        if split == "train":
            return self.train_seeds
        if split == "validation":
            return self.validation_seeds
        if split == "heldout":
            return self.heldout_seeds
        raise ValueError(f"unknown split {split!r}")


def load_policy(policy_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("hlbench_candidate_policy", policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy_cls = getattr(module, "Policy")
    return policy_cls()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run_dir(run_id: str | None, output_dir: Path | None = None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_episode(
    *,
    adapter: GymnasiumMinigridAdapter,
    policy: Any,
    seed: int,
    max_steps: int,
    task_config: dict[str, Any],
    replay_path: Path | None = None,
) -> dict[str, Any]:
    observation = adapter.reset(seed=seed, config=task_config)
    if hasattr(policy, "reset"):
        policy.reset(seed, task_config)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    invalid_actions = 0
    action_histogram: dict[str, int] = {}
    replay: list[dict[str, Any]] = []

    while steps < max_steps:
        obs_before = observation
        try:
            action = policy.act(observation, {"action_count": adapter.action_count})
        except Exception as exc:
            _append_replay_exception(
                replay,
                t=steps,
                obs_before=obs_before,
                error_type="policy_exception",
                exception=repr(exc),
            )
            _write_replay(replay_path, replay)
            return {
                "seed": seed,
                "score": total_reward,
                "steps": steps,
                "terminated": False,
                "truncated": False,
                "terminated_by": "policy_exception",
                "invalid_actions": invalid_actions,
                "exception": repr(exc),
                "action_histogram": action_histogram,
            }

        try:
            step_result = adapter.step(int(action))
        except Exception as exc:
            invalid_actions += 1
            _append_replay_invalid_action(
                replay,
                t=steps,
                obs_before=obs_before,
                action=action,
                exception=repr(exc),
            )
            _write_replay(replay_path, replay)
            return {
                "seed": seed,
                "score": total_reward,
                "steps": steps,
                "terminated": False,
                "truncated": False,
                "terminated_by": "invalid_action",
                "invalid_actions": invalid_actions,
                "exception": repr(exc),
                "action_histogram": action_histogram,
            }

        action_key = str(int(action))
        action_histogram[action_key] = action_histogram.get(action_key, 0) + 1
        _append_replay_step(
            replay,
            t=steps,
            obs_before=obs_before,
            action=int(action),
            reward=step_result.reward,
            obs_after=step_result.observation,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            info=step_result.info,
        )
        total_reward += step_result.reward
        steps += 1
        observation = step_result.observation
        terminated = step_result.terminated
        truncated = step_result.truncated
        if step_result.done:
            break

    if terminated and total_reward > 0:
        terminated_by = "success"
    elif terminated:
        terminated_by = "terminated"
    elif truncated or steps >= max_steps:
        terminated_by = "timeout"
    else:
        terminated_by = "unknown"

    _write_replay(replay_path, replay)
    return {
        "seed": seed,
        "score": total_reward,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated or steps >= max_steps,
        "terminated_by": terminated_by,
        "invalid_actions": invalid_actions,
        "action_histogram": action_histogram,
    }


def _append_replay_step(
    replay: list[dict[str, Any]],
    *,
    t: int,
    obs_before: dict[str, Any],
    action: int,
    reward: float,
    obs_after: dict[str, Any],
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> None:
    replay.append(
        {
            "t": t,
            "obs_before": obs_before,
            "action": action,
            "action_name": ACTION_NAMES.get(action, f"action_{action}"),
            "reward": reward,
            "obs_after": obs_after,
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
            "info": info,
        }
    )


def _append_replay_exception(
    replay: list[dict[str, Any]],
    *,
    t: int,
    obs_before: dict[str, Any],
    error_type: str,
    exception: str,
) -> None:
    replay.append(
        {
            "t": t,
            "obs_before": obs_before,
            "action": None,
            "action_name": None,
            "reward": 0.0,
            "obs_after": None,
            "terminated": False,
            "truncated": False,
            "done": True,
            "error_type": error_type,
            "exception": exception,
        }
    )


def _append_replay_invalid_action(
    replay: list[dict[str, Any]],
    *,
    t: int,
    obs_before: dict[str, Any],
    action: Any,
    exception: str,
) -> None:
    replay.append(
        {
            "t": t,
            "obs_before": obs_before,
            "action": action,
            "action_name": None,
            "reward": 0.0,
            "obs_after": None,
            "terminated": False,
            "truncated": False,
            "done": True,
            "error_type": "invalid_action",
            "exception": exception,
        }
    )


def _write_replay(path: Path | None, replay: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for step in replay:
            handle.write(json.dumps(step, sort_keys=True) + "\n")


def run_policy(
    *,
    scenario_name: str,
    split: str,
    policy_path: Path,
    episodes: int | None,
    run_id: str | None,
    output_dir: Path | None = None,
    write_replays: bool = True,
) -> Path:
    scenario = Scenario.load(scenario_name)
    split_seeds = scenario.seeds_for_split(split)
    if episodes is not None:
        split_seeds = split_seeds[:episodes]

    run_dir = make_run_dir(run_id, output_dir)
    rollout_dir = run_dir / "rollout"
    rollout_dir.mkdir(parents=True)
    if write_replays:
        replays_dir = rollout_dir / "replays"
        replays_dir.mkdir()

    config = {
        "scenario_name": scenario_name,
        "scenario_id": scenario.scenario_id,
        "env_id": scenario.env_id,
        "split": split,
        "seeds": split_seeds if split != "heldout" else "<hidden>",
        "episodes": len(split_seeds),
        "max_steps": scenario.max_steps,
        "policy_path": str(policy_path),
        "policy_sha256": file_sha256(policy_path),
        "write_replays": write_replays,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    adapter = GymnasiumMinigridAdapter(scenario.env_id, max_steps=scenario.max_steps)
    policy = load_policy(policy_path)
    trials = []
    try:
        for index, seed in enumerate(split_seeds):
            trial_id = f"trial_{index:06d}"
            replay_relative_path = f"replays/{trial_id}.jsonl" if write_replays else None
            trial = run_episode(
                adapter=adapter,
                policy=policy,
                seed=seed,
                max_steps=scenario.max_steps,
                task_config={
                    "scenario_id": scenario.scenario_id,
                    "env_id": scenario.env_id,
                    "max_steps": scenario.max_steps,
                },
                replay_path=(rollout_dir / replay_relative_path) if replay_relative_path else None,
            )
            trial["trial_id"] = trial_id
            trial["split"] = split
            if replay_relative_path:
                trial["replay_path"] = replay_relative_path
            trials.append(trial)
    finally:
        adapter.close()

    with (rollout_dir / "trials.jsonl").open("w") as handle:
        for trial in trials:
            handle.write(json.dumps(trial, sort_keys=True) + "\n")

    summary, failures = summarize_trials(
        scenario_id=scenario.scenario_id,
        env_id=scenario.env_id,
        split=split,
        trials=trials,
        policy_sha256=config["policy_sha256"],
    )
    (rollout_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (rollout_dir / "failures.jsonl").open("w") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, sort_keys=True) + "\n")

    print(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="minigrid_doorkey")
    parser.add_argument("--split", choices=["train", "validation", "heldout"], default="train")
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-replays", action="store_true")
    args = parser.parse_args()

    policy_path = args.policy
    if policy_path is None:
        policy_path = SCENARIO_ROOT / args.scenario / "policy.py"
    run_policy(
        scenario_name=args.scenario,
        split=args.split,
        policy_path=policy_path,
        episodes=args.episodes,
        run_id=args.run_id,
        output_dir=args.output_dir,
        write_replays=not args.no_replays,
    )


if __name__ == "__main__":
    main()
