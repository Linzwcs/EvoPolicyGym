"""One fresh MiniGrid Memory Environment per Episode."""

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

from .config import MemoryConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(3))
_MISSION = "go to the matching object at the end of the hallway"
_KEY = 5
_BALL = 6
_GREEN = 1
_ACTION_NAMES = ("turn_left", "turn_right", "move_forward")


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    green_key_visible: bool
    green_ball_visible: bool
    front_green_object: str


class MemoryEnvironment:
    """The seeded strict adapter around a configured MiniGrid Memory task."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: MemoryConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not MemoryConfig:
            raise TypeError("config must be MemoryConfig")
        if episode.scenario is not None:
            raise ValueError(
                "Memory configuration belongs in MemoryConfig, not EpisodeSpec.scenario"
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
        self._green_key_found = False
        self._green_ball_found = False
        self._green_key_first_seen_step = -1
        self._green_ball_first_seen_step = -1
        self._first_observed_green_object_type = "none"
        self._selected_object_type = "none"
        self._decision_step = -1
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
            raise RuntimeError("MiniGrid Memory returned an unexpected horizon")
        self._update_visible_objects(facts, step=0)
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
            raise RuntimeError("MiniGrid Memory returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid Memory observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        ineffective_action = signature == previous_signature and number == 0.0
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._ineffective_actions += int(ineffective_action)
        self._action_counts[action] += 1
        self._update_visible_objects(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        wrong_target = bool(terminated and number == 0.0)
        if terminated and not (success or wrong_target):
            raise RuntimeError("MiniGrid Memory termination semantics drifted")
        expected_reward = 1.0 - 0.9 * self._steps / self._max_steps if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid Memory reward semantics drifted")
        if truncated != (self._steps == self._max_steps):
            raise RuntimeError("MiniGrid Memory horizon semantics drifted")
        if success or wrong_target:
            self._decision_step = self._steps
            self._selected_object_type = _selected_object_type(facts)
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif wrong_target and truncated:
            terminal_reason = "wrong_target_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif wrong_target:
            terminal_reason = "wrong_target"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = "searching_for_first_green_object"
        if terminal_reason != "none":
            task_stage = terminal_reason
        elif self._green_key_found and self._green_ball_found:
            task_stage = "both_green_object_types_observed"
        elif self._green_key_found or self._green_ball_found:
            task_stage = "one_green_object_type_observed"
        self._done = terminated or truncated
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._max_steps - self._steps, 0),
            "green_key_visible": facts.green_key_visible,
            "green_key_found": self._green_key_found,
            "green_key_first_seen_step": self._green_key_first_seen_step,
            "green_ball_visible": facts.green_ball_visible,
            "green_ball_found": self._green_ball_found,
            "green_ball_first_seen_step": self._green_ball_first_seen_step,
            "visible_green_object_types": _visible_object_types(facts),
            "first_observed_green_object_type": (self._first_observed_green_object_type),
            "selected_object_type": self._selected_object_type,
            "decision_step": self._decision_step,
            "task_stage": task_stage,
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": (self._novel_observation_steps / self._steps),
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "success_reward_at_this_step": 1.0 - 0.9 * self._steps / self._max_steps,
            "cumulative_return": self._cumulative_return,
            "wrong_target": wrong_target,
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

    def _update_visible_objects(
        self,
        facts: _ObservationFacts,
        *,
        step: int,
    ) -> None:
        if facts.green_key_visible and self._green_key_first_seen_step < 0:
            self._green_key_first_seen_step = step
        if facts.green_ball_visible and self._green_ball_first_seen_step < 0:
            self._green_ball_first_seen_step = step
        if self._first_observed_green_object_type == "none":
            observed = _visible_object_types(facts)
            if observed != "none":
                self._first_observed_green_object_type = observed
        self._green_key_found = self._green_key_found or facts.green_key_visible
        self._green_ball_found = self._green_ball_found or facts.green_ball_visible


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid Memory returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Memory returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid Memory returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid Memory returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Memory returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid Memory returned an invalid mission")

    key_mask = (image[:, :, 0] == _KEY) & (image[:, :, 1] == _GREEN)
    ball_mask = (image[:, :, 0] == _BALL) & (image[:, :, 1] == _GREEN)
    front_object_code = int(image[3, 5, 0])
    front_color_code = int(image[3, 5, 1])
    front_green_object = "none"
    if front_color_code == _GREEN and front_object_code == _KEY:
        front_green_object = "key"
    elif front_color_code == _GREEN and front_object_code == _BALL:
        front_green_object = "ball"
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
            green_key_visible=bool(numpy.any(key_mask)),
            green_ball_visible=bool(numpy.any(ball_mask)),
            front_green_object=front_green_object,
        ),
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid Memory returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid Memory returned a non-finite {name}")
    return number


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid Memory public observation is invalid")
    return image.data, direction


def _visible_object_types(facts: _ObservationFacts) -> str:
    if facts.green_key_visible and facts.green_ball_visible:
        return "key_and_ball"
    if facts.green_key_visible:
        return "key"
    if facts.green_ball_visible:
        return "ball"
    return "none"


def _selected_object_type(facts: _ObservationFacts) -> str:
    if facts.front_green_object != "none":
        return facts.front_green_object
    visible = _visible_object_types(facts)
    return visible if visible in {"key", "ball"} else "unknown"


__all__ = ["MemoryEnvironment"]
