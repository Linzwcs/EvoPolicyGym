"""A parameterized LunarLander-v3 Benchmark with public traces."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Sequence

from evopolicygym.authoring import (
    Artifact,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    Feedback,
)
from evopolicygym.policy import PolicyValue

from .config import LunarLanderConfig
from .environment import LunarLanderEnvironment

_EPISODE_SEED_DOMAIN = b"evopolicygym-lunar-lander/episode-seed/v1\0"
_SPLITS = frozenset({"train", "validation", "test"})
_MAX_TRACED_EPISODES = 8
_MAX_EPISODE_STEPS = 1_000
_FAILURE_RETURN = -1_000.0
_OBSERVATION_FIELDS = (
    "x_position",
    "y_position",
    "x_velocity",
    "y_velocity",
    "angle",
    "angular_velocity",
    "left_leg_contact",
    "right_leg_contact",
)


class LunarLanderBenchmark:
    """Mean LunarLander return over deterministic Episode plans."""

    def __init__(self, config: LunarLanderConfig | None = None) -> None:
        if config is None:
            config = LunarLanderConfig()
        if type(config) is not LunarLanderConfig:
            raise TypeError("config must be LunarLanderConfig")
        self._config = config
        self._spec = _benchmark_spec(config)

    @property
    def spec(self) -> BenchmarkSpec:
        return self._spec

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        if type(split) is not str or split not in _SPLITS:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return tuple(
            EpisodeSpec(
                environment_seed=_episode_seed(split, seed, index),
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        return LunarLanderEnvironment(episode, config=self._config)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        records = tuple(episodes)
        if not records:
            raise ValueError("episodes must be non-empty")
        if any(type(record) is not EpisodeRecord for record in records):
            raise TypeError("episodes must contain EpisodeRecord values")

        returns = tuple(
            (
                record.total_reward
                if record.policy_failure is None
                else _FAILURE_RETURN
            )
            for record in records
        )
        score = statistics.fmean(returns)
        landings = sum(_landed(record) for record in records)
        failures = sum(record.policy_failure is not None for record in records)
        mean_steps = statistics.fmean(record.steps for record in records)
        traced = records[:_MAX_TRACED_EPISODES]
        return Feedback(
            score=score,
            content={
                "summary": (
                    f"Mean return {score:.3f} across {len(records)} Episodes; "
                    f"{landings} successful landings."
                ),
                "mean_return": score,
                "mean_steps": mean_steps,
                "episodes": len(records),
                "successful_landings": landings,
                "policy_failures": failures,
                "failure_return": _FAILURE_RETURN,
                "traced_episodes": len(traced),
                "trace_episodes_omitted": len(records) - len(traced),
            },
            artifacts=(
                _trace_artifact(
                    traced,
                    continuous=self._config.continuous,
                ),
            ),
        )


def _benchmark_spec(config: LunarLanderConfig) -> BenchmarkSpec:
    action_space: PolicyValue
    if config.continuous:
        action_space = {
            "type": "array",
            "shape": [2],
            "items": {
                "type": "float",
                "minimum": -1.0,
                "maximum": 1.0,
            },
            "components": [
                "main_engine",
                "lateral_engine",
            ],
            "notes": (
                "Main values <= 0 disable the main engine; positive values "
                "control throttle. Lateral values below -0.5 fire the left "
                "orientation engine and values above 0.5 fire the right."
            ),
        }
    else:
        action_space = {
            "type": "discrete",
            "values": [0, 1, 2, 3],
            "meaning": {
                "0": "do_nothing",
                "1": "fire_left_orientation_engine",
                "2": "fire_main_engine",
                "3": "fire_right_orientation_engine",
            },
        }
    return BenchmarkSpec(
        id="gymnasium/LunarLander-v3/mean-return-v1",
        description=(
            "Control a lunar lander from a randomized initial impulse to a "
            "safe landing pad. Balance position, velocity, attitude, landing "
            "contact, and engine cost. Maximize mean Episode return."
        ),
        observation_space={
            "type": "object",
            "fields": {
                "x_position": {
                    "type": "float",
                    "minimum": -2.5,
                    "maximum": 2.5,
                },
                "y_position": {
                    "type": "float",
                    "minimum": -2.5,
                    "maximum": 2.5,
                },
                "x_velocity": {
                    "type": "float",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
                "y_velocity": {
                    "type": "float",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
                "angle": {
                    "type": "float",
                    "minimum": -6.283185307179586,
                    "maximum": 6.283185307179586,
                },
                "angular_velocity": {
                    "type": "float",
                    "minimum": -10.0,
                    "maximum": 10.0,
                },
                "left_leg_contact": {"type": "boolean"},
                "right_leg_contact": {"type": "boolean"},
            },
        },
        action_space=action_space,
        metadata={
            "environment": "LunarLander-v3",
            "provider": "Gymnasium",
            "reward_threshold": 200.0,
            "successful_landing_terminal_reward": 100.0,
            "crash_terminal_reward": -100.0,
            "failure_return": _FAILURE_RETURN,
        },
        environment_parameters={
            "continuous": config.continuous,
            "gravity": config.gravity,
            "enable_wind": config.enable_wind,
            "wind_power": config.wind_power,
            "turbulence_power": config.turbulence_power,
        },
        max_episode_steps=_MAX_EPISODE_STEPS,
        primary_metric="mean_return",
        score_direction="maximize",
    )


def _episode_seed(split: str, seed: int, index: int) -> int:
    digest = hashlib.sha256()
    digest.update(_EPISODE_SEED_DOMAIN)
    digest.update(split.encode("ascii"))
    digest.update(b"\0")
    digest.update(seed.to_bytes(8, "big"))
    digest.update(index.to_bytes(8, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


def _landed(record: EpisodeRecord) -> bool:
    return bool(
        record.policy_failure is None
        and record.transitions
        and record.transitions[-1].step.terminated
        and record.transitions[-1].step.reward == 100.0
    )


def _trace_artifact(
    records: Sequence[EpisodeRecord],
    *,
    continuous: bool,
) -> Artifact:
    lines: list[bytes] = []
    for episode_index, record in enumerate(records):
        lines.append(
            _json_line(
                {
                    "type": "episode",
                    "episode_index": episode_index,
                    "status": (
                        "completed"
                        if record.policy_failure is None
                        else "policy_failed"
                    ),
                    "steps": record.steps,
                    "return": (
                        record.total_reward
                        if record.policy_failure is None
                        else _FAILURE_RETURN
                    ),
                    "landed": _landed(record),
                    "failure": record.policy_failure,
                }
            )
        )
        observation = _trace_observation(record.initial_observation)
        for step_index, transition in enumerate(record.transitions):
            action = _trace_action(
                transition.action,
                continuous=continuous,
            )
            next_observation = _trace_observation(
                transition.step.observation
            )
            lines.append(
                _json_line(
                    {
                        "type": "transition",
                        "episode_index": episode_index,
                        "step_index": step_index,
                        "observation": observation,
                        "action": action,
                        "reward": transition.step.reward,
                        "next_observation": next_observation,
                        "terminated": transition.step.terminated,
                        "truncated": transition.step.truncated,
                    }
                )
            )
            observation = next_observation
    return Artifact(
        name="trace.jsonl",
        media_type="application/x-ndjson",
        content=b"".join(lines),
    )


def _trace_action(
    action: PolicyValue,
    *,
    continuous: bool,
) -> PolicyValue:
    if continuous:
        if (
            type(action) is not list
            or len(action) != 2
            or any(
                type(value) is not float
                or not -1.0 <= value <= 1.0
                for value in action
            )
        ):
            raise ValueError("LunarLander trace Action is invalid")
        return list(action)
    if type(action) is not int or action not in {0, 1, 2, 3}:
        raise ValueError("LunarLander trace Action is invalid")
    return action


def _trace_observation(
    observation: PolicyValue,
) -> dict[str, PolicyValue]:
    if type(observation) is not dict:
        raise ValueError("LunarLander trace observation is invalid")
    if set(observation) != set(_OBSERVATION_FIELDS):
        raise ValueError("LunarLander trace observation is invalid")
    traced: dict[str, PolicyValue] = {}
    for key in _OBSERVATION_FIELDS[:6]:
        value = observation[key]
        if type(value) is not float:
            raise ValueError("LunarLander trace observation is invalid")
        traced[key] = value
    for key in _OBSERVATION_FIELDS[6:]:
        value = observation[key]
        if type(value) is not bool:
            raise ValueError("LunarLander trace observation is invalid")
        traced[key] = value
    return traced


def _json_line(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")


__all__ = ["LunarLanderBenchmark"]
