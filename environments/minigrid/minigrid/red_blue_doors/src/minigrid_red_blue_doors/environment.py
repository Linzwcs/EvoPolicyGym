"""One fresh MiniGrid RedBlueDoors Environment per Episode."""

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

from .config import RedBlueDoorsConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "open the red door then the blue door"
_DOOR = 4
_RED = 0
_BLUE = 2
_OPEN = 0
_CLOSED = 1
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
class _ObservationFacts:
    red_visible: bool
    blue_visible: bool
    front_door: tuple[int, int] | None


class RedBlueDoorsEnvironment:
    """Strict seeded adapter around MiniGrid RedBlueDoors."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: RedBlueDoorsConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not RedBlueDoorsConfig:
            raise TypeError("config must be RedBlueDoorsConfig")
        if episode.scenario is not None:
            raise ValueError(
                "RedBlueDoors configuration belongs in RedBlueDoorsConfig, not EpisodeSpec.scenario"
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
        self._front_door: tuple[int, int] | None = None
        self._red_found = False
        self._blue_found = False
        self._red_ever_opened = False
        self._red_is_open = False
        self._red_reclosed = False
        self._blue_opened = False
        self._red_first_seen_step = -1
        self._blue_first_seen_step = -1
        self._red_open_step = -1
        self._red_reclose_step = -1
        self._blue_open_step = -1
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
            raise RuntimeError("MiniGrid RedBlueDoors returned an unexpected horizon")
        self._update(facts)
        self._red_first_seen_step = 0 if facts.red_visible else -1
        self._blue_first_seen_step = 0 if facts.blue_visible else -1
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
        red_was_open = self._red_is_open
        red_was_ever_opened = self._red_ever_opened
        front_door_before_action = _front_door_label(self._front_door)
        opened_red = bool(action == 5 and self._front_door == (_RED, _CLOSED))
        closed_red = bool(action == 5 and self._front_door == (_RED, _OPEN))
        opened_blue = bool(action == 5 and self._front_door == (_BLUE, _CLOSED))
        effective_toggle = opened_red or closed_red or opened_blue
        opened_blue_before_red = bool(opened_blue and not red_was_open and not red_was_ever_opened)
        opened_blue_after_red_reclosed = bool(
            opened_blue and not red_was_open and red_was_ever_opened
        )
        order_error = opened_blue_before_red or opened_blue_after_red_reclosed
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid RedBlueDoors returned invalid termination flags")
        number = _number(reward)
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid RedBlueDoors observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        ineffective_action = signature == previous_signature and number == 0.0
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._ineffective_actions += int(ineffective_action)
        self._action_counts[action] += 1
        if facts.red_visible and self._red_first_seen_step < 0:
            self._red_first_seen_step = self._steps
        if facts.blue_visible and self._blue_first_seen_step < 0:
            self._blue_first_seen_step = self._steps
        if opened_red and self._red_open_step < 0:
            self._red_open_step = self._steps
        if closed_red and self._red_reclose_step < 0:
            self._red_reclose_step = self._steps
        if opened_blue and self._blue_open_step < 0:
            self._blue_open_step = self._steps
        if opened_red:
            self._red_is_open = True
            self._red_ever_opened = True
        elif closed_red:
            self._red_is_open = False
            self._red_reclosed = True
        self._blue_opened = self._blue_opened or opened_blue
        self._update(facts)
        success = bool(terminated and number > 0.0)
        if success != (opened_blue and red_was_open):
            raise RuntimeError("MiniGrid RedBlueDoors success semantics drifted")
        expected_reward = 1.0 - 0.9 * self._steps / self._max_steps if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid RedBlueDoors reward semantics drifted")
        if terminated != (success or order_error):
            raise RuntimeError("MiniGrid RedBlueDoors termination semantics drifted")
        if truncated != (self._steps == self._max_steps):
            raise RuntimeError("MiniGrid RedBlueDoors horizon semantics drifted")
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif opened_blue_before_red and truncated:
            terminal_reason = "blue_before_red_and_time_limit"
        elif opened_blue_after_red_reclosed and truncated:
            terminal_reason = "red_reclosed_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif opened_blue_before_red:
            terminal_reason = "blue_before_red"
        elif opened_blue_after_red_reclosed:
            terminal_reason = "red_reclosed_before_blue"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = "find_red"
        if terminal_reason != "none":
            task_stage = terminal_reason
        elif not self._red_ever_opened:
            task_stage = "open_red" if self._red_found else "find_red"
        elif not self._red_is_open:
            task_stage = "reopen_red"
        elif not self._blue_found:
            task_stage = "find_blue"
        else:
            task_stage = "open_blue"
        self._done = terminated or truncated
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._max_steps - self._steps, 0),
            "red_door_visible": facts.red_visible,
            "red_door_found": self._red_found,
            "red_door_first_seen_step": self._red_first_seen_step,
            "blue_door_visible": facts.blue_visible,
            "blue_door_found": self._blue_found,
            "blue_door_first_seen_step": self._blue_first_seen_step,
            "front_door_before_action": front_door_before_action,
            "effective_toggle": effective_toggle,
            "red_door_opened_this_step": opened_red,
            "red_door_opened": self._red_ever_opened,
            "red_door_open_step": self._red_open_step,
            "red_door_open": self._red_is_open,
            "red_door_closed_this_step": closed_red,
            "red_door_reclosed": self._red_reclosed,
            "red_door_reclose_step": self._red_reclose_step,
            "blue_door_opened_this_step": opened_blue,
            "blue_door_opened": self._blue_opened,
            "blue_door_open_step": self._blue_open_step,
            "blue_opened_before_red": opened_blue_before_red,
            "blue_opened_after_red_reclosed": opened_blue_after_red_reclosed,
            "order_error": order_error,
            "task_stage": task_stage,
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

    def _update(self, facts: _ObservationFacts) -> None:
        self._front_door = facts.front_door
        self._red_found = self._red_found or facts.red_visible
        self._blue_found = self._blue_found or facts.blue_visible


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid RedBlueDoors returned an invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid RedBlueDoors returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid RedBlueDoors returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid RedBlueDoors returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid RedBlueDoors returned an invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid RedBlueDoors returned an invalid mission")
    front_door = (int(image[3, 5, 1]), int(image[3, 5, 2])) if image[3, 5, 0] == _DOOR else None
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
            red_visible=bool(numpy.any((image[:, :, 0] == _DOOR) & (image[:, :, 1] == _RED))),
            blue_visible=bool(numpy.any((image[:, :, 0] == _DOOR) & (image[:, :, 1] == _BLUE))),
            front_door=front_door,
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid RedBlueDoors returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid RedBlueDoors returned non-finite reward")
    return number


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid RedBlueDoors public observation is invalid")
    return image.data, direction


def _front_door_label(front_door: tuple[int, int] | None) -> str:
    if front_door is None:
        return "none"
    color, state = front_door
    color_name = "red" if color == _RED else "blue"
    state_name = "open" if state == _OPEN else "closed"
    return f"{color_name}_{state_name}"


__all__ = ["RedBlueDoorsEnvironment"]
