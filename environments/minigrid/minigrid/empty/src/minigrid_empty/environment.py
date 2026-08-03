"""One fresh MiniGrid Empty Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import EmptyConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_GOAL = 8
_ACTION_NAMES = (
    "turn_left",
    "turn_right",
    "move_forward",
    "pick_up",
    "drop",
    "toggle",
    "done",
)


@dataclass(frozen=True, slots=True)
class _Facts:
    goal_visible: bool


class EmptyEnvironment:
    """Strict seeded adapter around a configured Empty registration."""

    def __init__(self, episode: EpisodeSpec, *, config: EmptyConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not EmptyConfig:
            raise TypeError("config must be EmptyConfig")
        if episode.scenario is not None:
            raise ValueError("Empty configuration belongs in EmptyConfig")
        self._seed = episode.environment_seed
        self._max_steps = config.max_episode_steps
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._goal_found = False
        self._goal_first_seen_step = -1
        self._steps = 0
        self._previous_observation_signature: tuple[bytes, int] | None = None
        self._seen_observation_signatures: set[tuple[bytes, int]] = set()
        self._novel_observation_steps = 0
        self._ineffective_actions = 0
        self._action_counts = [0] * len(_ACTION_NAMES)
        self._cumulative_return = 0.0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public, facts = _observation(observation)
        horizon = self._environment.get_wrapper_attr("max_steps")
        if type(horizon) is not int or horizon != self._max_steps:
            raise RuntimeError("MiniGrid Empty returned an unexpected horizon")
        self._goal_found = facts.goal_visible
        self._goal_first_seen_step = 0 if facts.goal_visible else -1
        signature = _observation_signature(public)
        self._previous_observation_signature = signature
        self._seen_observation_signatures.add(signature)
        self._started = True
        return public

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not int or action not in _ACTIONS:
            raise InvalidAction()
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid Empty returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid Empty observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        ineffective_action = signature == previous_signature and number == 0.0
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._ineffective_actions += int(ineffective_action)
        self._action_counts[action] += 1
        if facts.goal_visible and self._goal_first_seen_step < 0:
            self._goal_first_seen_step = self._steps
        self._goal_found = self._goal_found or facts.goal_visible
        success = bool(terminated and number > 0.0)
        expected_reward = 1.0 - 0.9 * self._steps / self._max_steps if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid Empty reward semantics drifted")
        if terminated != success:
            raise RuntimeError("MiniGrid Empty termination semantics drifted")
        if truncated != (self._steps == self._max_steps):
            raise RuntimeError("MiniGrid Empty horizon semantics drifted")
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif truncated:
            terminal_reason = "time_limit"
        self._done = terminated or truncated
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._max_steps - self._steps, 0),
            "goal_visible": facts.goal_visible,
            "goal_found": self._goal_found,
            "goal_first_seen_step": self._goal_first_seen_step,
            "steps_since_goal_first_seen": (
                self._steps - self._goal_first_seen_step if self._goal_first_seen_step >= 0 else -1
            ),
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": (self._novel_observation_steps / self._steps),
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "success_reward_at_this_step": 1.0 - 0.9 * self._steps / self._max_steps,
            "cumulative_return": self._cumulative_return,
            "success": success,
            "terminal_reason": terminal_reason,
        }
        for name, count in zip(_ACTION_NAMES, self._action_counts, strict=True):
            metrics[f"{name}_count"] = count
        return Step(
            observation=public,
            reward=number,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid Empty returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Empty returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid Empty returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Empty returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != "get to the green goal square":
        raise RuntimeError("MiniGrid Empty returned invalid mission")
    return (
        {
            "image": TensorValue(
                dtype="uint8",
                shape=_IMAGE_SHAPE,
                data=image.tobytes(order="C"),
            ),
            "direction": direction,
            "mission": mission,
        },
        _Facts(goal_visible=bool(numpy.any(image[:, :, 0] == _GOAL))),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid Empty returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid Empty returned non-finite reward")
    return number


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid Empty public observation is invalid")
    return image.data, direction


__all__ = ["EmptyEnvironment"]
