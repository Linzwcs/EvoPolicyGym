"""One fresh MiniGrid DynamicObstacles Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401  # Import registers MiniGrid environments.
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from .config import DynamicObstaclesConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(3))
_MISSION = "get to the green goal square"
_GOAL = 8
_BALL = 6
_WALL = 2
_OBJECT_NAMES = (
    "unseen",
    "empty",
    "wall",
    "floor",
    "door",
    "key",
    "ball",
    "box",
    "goal",
    "lava",
    "agent",
)
_ACTION_NAMES = ("turn_left", "turn_right", "move_forward")


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    goal_visible: bool
    visible_obstacle_count: int


class DynamicObstaclesEnvironment:
    """Strict seeded adapter around MiniGrid DynamicObstacles."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: DynamicObstaclesConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not DynamicObstaclesConfig:
            raise TypeError("config must be DynamicObstaclesConfig")
        if episode.scenario is not None:
            raise ValueError(
                "DynamicObstacles configuration belongs in "
                "DynamicObstaclesConfig, not EpisodeSpec.scenario"
            )

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
        self._obstacle_found = False
        self._goal_first_seen_step = -1
        self._obstacle_first_seen_step = -1
        self._steps = 0
        self._previous_observation_signature: tuple[bytes, int] | None = None
        self._front_object_code = -1
        self._seen_observation_signatures: set[tuple[bytes, int]] = set()
        self._novel_observation_steps = 0
        self._ineffective_actions = 0
        self._obstacle_exposure_steps = 0
        self._max_visible_obstacle_count = 0
        self._action_counts = [0] * len(_ACTION_NAMES)
        self._collision_step = -1
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
            raise RuntimeError("MiniGrid DynamicObstacles returned an unexpected horizon")
        self._goal_found = facts.goal_visible
        self._obstacle_found = facts.visible_obstacle_count > 0
        self._goal_first_seen_step = 0 if facts.goal_visible else -1
        self._obstacle_first_seen_step = 0 if facts.visible_obstacle_count > 0 else -1
        self._max_visible_obstacle_count = facts.visible_obstacle_count
        signature = _observation_signature(public)
        self._previous_observation_signature = signature
        self._front_object_code = _front_object_code(public)
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
            raise RuntimeError("MiniGrid DynamicObstacles returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid DynamicObstacles observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        ineffective_action = signature == previous_signature and number == 0.0
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._ineffective_actions += int(ineffective_action)
        obstacle_visible = facts.visible_obstacle_count > 0
        self._obstacle_exposure_steps += int(obstacle_visible)
        self._max_visible_obstacle_count = max(
            self._max_visible_obstacle_count,
            facts.visible_obstacle_count,
        )
        self._action_counts[action] += 1
        if facts.goal_visible and self._goal_first_seen_step < 0:
            self._goal_first_seen_step = self._steps
        if obstacle_visible and self._obstacle_first_seen_step < 0:
            self._obstacle_first_seen_step = self._steps
        self._goal_found = self._goal_found or facts.goal_visible
        self._obstacle_found = self._obstacle_found or obstacle_visible
        collided = bool(terminated and number == -1.0)
        success = bool(terminated and number > 0.0)
        if collided and action != 2:
            raise RuntimeError("MiniGrid DynamicObstacles collision semantics drifted")
        obstacle_collision = bool(collided and self._front_object_code == _BALL)
        wall_collision = bool(collided and self._front_object_code == _WALL)
        blocked_by = "none"
        if obstacle_collision:
            blocked_by = "moving_obstacle"
        elif wall_collision:
            blocked_by = "wall"
        elif collided:
            blocked_by = "other"
        expected_reward = -1.0 if collided else 0.0
        if success:
            expected_reward = 1.0 - 0.9 * self._steps / self._max_steps
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid DynamicObstacles reward semantics drifted")
        if terminated != (success or collided):
            raise RuntimeError("MiniGrid DynamicObstacles termination semantics drifted")
        if truncated != (self._steps == self._max_steps):
            raise RuntimeError("MiniGrid DynamicObstacles horizon semantics drifted")
        if collided:
            self._collision_step = self._steps
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif obstacle_collision and truncated:
            terminal_reason = "obstacle_collision_and_time_limit"
        elif wall_collision and truncated:
            terminal_reason = "wall_collision_and_time_limit"
        elif collided and truncated:
            terminal_reason = "blocked_forward_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif obstacle_collision:
            terminal_reason = "obstacle_collision"
        elif wall_collision:
            terminal_reason = "wall_collision"
        elif collided:
            terminal_reason = "blocked_forward"
        elif truncated:
            terminal_reason = "time_limit"
        self._done = terminated or truncated
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._max_steps - self._steps, 0),
            "goal_visible": facts.goal_visible,
            "goal_found": self._goal_found,
            "goal_first_seen_step": self._goal_first_seen_step,
            "obstacle_visible": obstacle_visible,
            "obstacle_found": self._obstacle_found,
            "obstacle_first_seen_step": self._obstacle_first_seen_step,
            "visible_obstacle_count": facts.visible_obstacle_count,
            "max_visible_obstacle_count": self._max_visible_obstacle_count,
            "obstacle_exposure_step_fraction": (self._obstacle_exposure_steps / self._steps),
            "front_object_before_action": _OBJECT_NAMES[self._front_object_code],
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": (self._novel_observation_steps / self._steps),
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "success_reward_at_this_step": 1.0 - 0.9 * self._steps / self._max_steps,
            "cumulative_return": self._cumulative_return,
            "collision": collided,
            "collision_step": self._collision_step,
            "obstacle_collision": obstacle_collision,
            "wall_collision": wall_collision,
            "blocked_by": blocked_by,
            "success": success,
            "terminal_reason": terminal_reason,
        }
        for name, count in zip(_ACTION_NAMES, self._action_counts, strict=True):
            metrics[f"{name}_count"] = count
        self._front_object_code = _front_object_code(public)
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


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid DynamicObstacles returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid DynamicObstacles returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid DynamicObstacles returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid DynamicObstacles returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid DynamicObstacles returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid DynamicObstacles returned an invalid mission")

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
        _ObservationFacts(
            goal_visible=bool(numpy.any(image[:, :, 0] == _GOAL)),
            visible_obstacle_count=int(numpy.count_nonzero(image[:, :, 0] == _BALL)),
        ),
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid DynamicObstacles returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid DynamicObstacles returned a non-finite {name}")
    return number


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid DynamicObstacles public observation is invalid")
    return image.data, direction


def _front_object_code(observation: dict[str, PolicyValue]) -> int:
    image = observation.get("image")
    if type(image) is not TensorValue:
        raise RuntimeError("MiniGrid DynamicObstacles public observation is invalid")
    return image.data[(3 * 7 + 5) * 3]


__all__ = ["DynamicObstaclesEnvironment"]
