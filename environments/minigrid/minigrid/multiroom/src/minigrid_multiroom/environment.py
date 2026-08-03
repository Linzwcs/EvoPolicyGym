"""One fresh MiniGrid MultiRoom Environment per Episode."""

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

from .config import MultiRoomConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "traverse the rooms to get to the goal"
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_OBJECTS = (
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
_STATES = ("open", "closed", "locked")
_DOOR = 4
_GOAL = 8
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
_UNUSED_ACTIONS = frozenset({3, 4, 6})


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    goal_visible: bool
    goal_in_front: bool
    visible_door_count: int
    visible_closed_door_count: int
    visible_open_door_count: int
    front_object: tuple[int, int, int]
    front_label: str


class MultiRoomEnvironment:
    """The seeded strict adapter around configured MiniGrid MultiRoom."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: MultiRoomConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not MultiRoomConfig:
            raise TypeError("config must be MultiRoomConfig")
        if episode.scenario is not None:
            raise ValueError(
                "MultiRoom configuration belongs in MultiRoomConfig, not EpisodeSpec.scenario"
            )

        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._previous_facts: _ObservationFacts | None = None
        self._steps = 0
        self._door_found = False
        self._door_first_seen_step = -1
        self._first_door_opened_step = -1
        self._door_open_events = 0
        self._door_close_events = 0
        self._door_crossing_events = 0
        self._goal_found = False
        self._goal_first_seen_step = -1
        self._toggle_attempts = 0
        self._failed_toggle_attempts = 0
        self._forward_attempts = 0
        self._blocked_forward_attempts = 0
        self._unused_actions = 0
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
        if type(horizon) is not int or horizon != self._config.max_episode_steps:
            raise RuntimeError("MiniGrid MultiRoom returned an unexpected horizon")
        self._update_discovery(facts, step=0)
        self._previous_facts = facts
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

        previous_facts = self._previous_facts
        previous_signature = self._previous_observation_signature
        if previous_facts is None or previous_signature is None:
            raise RuntimeError("MiniGrid MultiRoom observation history is unavailable")
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label
        goal_in_front_before_action = previous_facts.goal_in_front

        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid MultiRoom returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        toggle_attempt = action == 5
        door_state_changed = bool(
            toggle_attempt
            and front_before_action[0] == _DOOR
            and facts.front_object[0] == _DOOR
            and facts.front_object[2] != front_before_action[2]
        )
        door_opened = bool(
            door_state_changed
            and front_before_action[2] == _CLOSED
            and facts.front_object[2] == _OPEN
        )
        door_closed = bool(
            door_state_changed
            and front_before_action[2] == _OPEN
            and facts.front_object[2] == _CLOSED
        )
        failed_toggle = toggle_attempt and not door_state_changed
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        if door_opened:
            self._door_open_events += 1
            if self._first_door_opened_step < 0:
                self._first_door_opened_step = self._steps
        self._door_close_events += int(door_closed)

        forward_attempt = action == 2
        blocked_forward = bool(forward_attempt and signature == previous_signature)
        door_crossed = bool(
            forward_attempt
            and front_before_action[0] == _DOOR
            and front_before_action[2] == _OPEN
            and not blocked_forward
        )
        self._forward_attempts += int(forward_attempt)
        self._blocked_forward_attempts += int(blocked_forward)
        self._door_crossing_events += int(door_crossed)
        unused_action = action in _UNUSED_ACTIONS
        self._unused_actions += int(unused_action)

        self._update_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != (goal_in_front_before_action and forward_attempt):
            raise RuntimeError("MiniGrid MultiRoom goal semantics drifted")
        if terminated != success:
            raise RuntimeError("MiniGrid MultiRoom termination semantics drifted")
        horizon = self._config.max_episode_steps
        expected_reward = 1.0 - 0.9 * self._steps / horizon if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid MultiRoom reward semantics drifted")
        if truncated != (self._steps == horizon):
            raise RuntimeError("MiniGrid MultiRoom horizon semantics drifted")

        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated
        )
        self._ineffective_actions += int(ineffective_action)
        self._cumulative_return += number
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = self._task_stage(facts, terminal_reason=terminal_reason)
        self._done = terminated or truncated
        self._previous_facts = facts

        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(horizon - self._steps, 0),
            "required_connecting_door_count": self._config.maximum_rooms - 1,
            "visible_door_count": facts.visible_door_count,
            "visible_closed_door_count": facts.visible_closed_door_count,
            "visible_open_door_count": facts.visible_open_door_count,
            "door_visible": facts.visible_door_count > 0,
            "door_found": self._door_found,
            "door_first_seen_step": self._door_first_seen_step,
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "closed_door_in_front": _is_door_state(facts.front_object, _CLOSED),
            "closed_door_in_front_before_action": _is_door_state(front_before_action, _CLOSED),
            "open_door_in_front": _is_door_state(facts.front_object, _OPEN),
            "open_door_in_front_before_action": _is_door_state(front_before_action, _OPEN),
            "toggle_attempt": toggle_attempt,
            "toggle_effective": door_state_changed,
            "toggle_attempt_count": self._toggle_attempts,
            "failed_toggle": failed_toggle,
            "failed_toggle_count": self._failed_toggle_attempts,
            "door_opened_this_step": door_opened,
            "door_closed_this_step": door_closed,
            "door_open_event_count": self._door_open_events,
            "door_close_event_count": self._door_close_events,
            "first_door_opened_step": self._first_door_opened_step,
            "door_crossed_this_step": door_crossed,
            "door_crossing_event_count": self._door_crossing_events,
            "goal_visible": facts.goal_visible,
            "goal_found": self._goal_found,
            "goal_first_seen_step": self._goal_first_seen_step,
            "goal_in_front": facts.goal_in_front,
            "goal_in_front_before_action": goal_in_front_before_action,
            "forward_attempt": forward_attempt,
            "forward_attempt_count": self._forward_attempts,
            "blocked_forward": blocked_forward,
            "blocked_forward_count": self._blocked_forward_attempts,
            "unused_action": unused_action,
            "unused_action_count": self._unused_actions,
            "task_stage": task_stage,
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": self._novel_observation_steps / self._steps,
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "success_reward_at_this_step": 1.0 - 0.9 * self._steps / horizon,
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

    def _update_discovery(self, facts: _ObservationFacts, *, step: int) -> None:
        door_visible = facts.visible_door_count > 0
        if door_visible and not self._door_found:
            self._door_first_seen_step = step
        if facts.goal_visible and not self._goal_found:
            self._goal_first_seen_step = step
        self._door_found = self._door_found or door_visible
        self._goal_found = self._goal_found or facts.goal_visible

    def _task_stage(
        self,
        facts: _ObservationFacts,
        *,
        terminal_reason: str,
    ) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if facts.goal_in_front:
            return "enter_goal"
        if self._goal_found:
            return "approach_goal"
        if _is_door_state(facts.front_object, _CLOSED):
            return "open_door"
        if self._door_open_events > self._door_crossing_events:
            return "pass_open_door"
        if self._door_open_events:
            return "explore_next_room"
        if self._door_found:
            return "approach_door"
        return "explore_first_room"


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid MultiRoom returned an invalid observation")

    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid MultiRoom returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid MultiRoom returned out-of-range image codes")

    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid MultiRoom returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid MultiRoom returned an invalid direction")

    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid MultiRoom returned an invalid mission")

    objects = image[:, :, 0]
    states = image[:, :, 2]
    door_mask = objects == _DOOR
    front_object = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
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
            goal_visible=bool(numpy.any(objects == _GOAL)),
            goal_in_front=front_object[0] == _GOAL,
            visible_door_count=int(numpy.count_nonzero(door_mask)),
            visible_closed_door_count=int(numpy.count_nonzero(door_mask & (states == _CLOSED))),
            visible_open_door_count=int(numpy.count_nonzero(door_mask & (states == _OPEN))),
            front_object=front_object,
            front_label=_object_label(*front_object),
        ),
    )


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid MultiRoom returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid MultiRoom returned a non-finite {name}")
    return number


def _is_door_state(value: tuple[int, int, int], state: int) -> bool:
    return value[0] == _DOOR and value[2] == state


def _object_label(object_code: int, color_code: int, state_code: int) -> str:
    if not 0 <= object_code < len(_OBJECTS):
        return "unknown"
    if object_code == 0:
        return "unseen"
    if object_code == 1:
        return "empty"
    if not 0 <= color_code < len(_COLORS):
        return "unknown"
    if object_code == _DOOR:
        if not 0 <= state_code < len(_STATES):
            return "unknown"
        return f"{_COLORS[color_code]}_{_STATES[state_code]}_door"
    return f"{_COLORS[color_code]}_{_OBJECTS[object_code]}"


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid MultiRoom public observation is invalid")
    return image.data, direction


__all__ = ["MultiRoomEnvironment"]
