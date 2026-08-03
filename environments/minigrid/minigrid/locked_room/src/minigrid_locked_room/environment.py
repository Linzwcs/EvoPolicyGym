"""One fresh MiniGrid LockedRoom Environment per Episode."""

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

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
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
_KEY = 5
_GOAL = 8
_OPEN = 0
_CLOSED = 1
_LOCKED = 2
_MAX_STEPS = 190
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
    target_color: int
    key_room_color: int
    target_key_visible: bool
    carried_key_color: int | None
    goal_visible: bool
    goal_in_front: bool
    target_door_visible: bool
    visible_doors: tuple[tuple[int, int], ...]
    front_object: tuple[int, int, int]
    front_label: str


class LockedRoomEnvironment:
    """Strict seeded adapter around MiniGrid LockedRoom."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("LockedRoom has no Episode scenario overrides")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make("MiniGrid-LockedRoom-v0"),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target_color: int | None = None
        self._key_room_color: int | None = None
        self._previous_facts: _Facts | None = None
        self._key_found = False
        self._key_first_seen_step = -1
        self._key_picked_up = False
        self._key_picked_up_step = -1
        self._key_dropped = False
        self._key_room_door_opened = False
        self._key_room_door_opened_step = -1
        self._target_door_found = False
        self._target_door_first_seen_step = -1
        self._target_door_opened = False
        self._target_door_opened_step = -1
        self._goal_found = False
        self._goal_first_seen_step = -1
        self._opened_door_colors: set[int] = set()
        self._door_open_events = 0
        self._door_close_events = 0
        self._steps = 0
        self._pickup_attempts = 0
        self._failed_pickup_attempts = 0
        self._drop_attempts = 0
        self._failed_drop_attempts = 0
        self._toggle_attempts = 0
        self._failed_toggle_attempts = 0
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
        if type(horizon) is not int or horizon != _MAX_STEPS:
            raise RuntimeError("MiniGrid LockedRoom returned an unexpected horizon")
        self._target_color = facts.target_color
        self._key_room_color = facts.key_room_color
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
        if previous_facts is None:
            raise RuntimeError("MiniGrid LockedRoom observation history is unavailable")
        carried_before_action = previous_facts.carried_key_color
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label
        goal_in_front_before_action = previous_facts.goal_in_front
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid LockedRoom returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        if facts.target_color != self._target_color or facts.key_room_color != self._key_room_color:
            raise RuntimeError("MiniGrid LockedRoom changed mission target")
        self._steps += 1
        signature = _observation_signature(public)
        previous_signature = self._previous_observation_signature
        if previous_signature is None:
            raise RuntimeError("MiniGrid LockedRoom observation history is unavailable")
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        pickup_attempt = action == 3
        key_picked_up = bool(
            pickup_attempt and carried_before_action is None and facts.carried_key_color is not None
        )
        failed_pickup = pickup_attempt and not key_picked_up
        self._pickup_attempts += int(pickup_attempt)
        self._failed_pickup_attempts += int(failed_pickup)
        drop_attempt = action == 4
        key_dropped = bool(
            drop_attempt and carried_before_action is not None and facts.carried_key_color is None
        )
        failed_drop = drop_attempt and not key_dropped
        self._drop_attempts += int(drop_attempt)
        self._failed_drop_attempts += int(failed_drop)

        toggle_attempt = action == 5
        door_state_changed = bool(
            toggle_attempt
            and front_before_action[0] == _DOOR
            and facts.front_object[0] == _DOOR
            and facts.front_object[2] != front_before_action[2]
        )
        door_opened = bool(
            door_state_changed
            and front_before_action[2] in {_CLOSED, _LOCKED}
            and facts.front_object[2] == _OPEN
        )
        door_closed = bool(
            door_state_changed
            and front_before_action[2] == _OPEN
            and facts.front_object[2] == _CLOSED
        )
        target_door_opened = bool(
            door_opened
            and front_before_action[1] == facts.target_color
            and front_before_action[2] == _LOCKED
        )
        key_room_door_opened = bool(door_opened and front_before_action[1] == facts.key_room_color)
        failed_toggle = toggle_attempt and not door_state_changed
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        if key_picked_up and facts.carried_key_color != facts.target_color:
            raise RuntimeError("MiniGrid LockedRoom key pickup semantics drifted")
        if target_door_opened and carried_before_action != facts.target_color:
            raise RuntimeError("MiniGrid LockedRoom target door semantics drifted")
        if door_opened:
            self._door_open_events += 1
            self._opened_door_colors.add(front_before_action[1])
        self._door_close_events += int(door_closed)
        if key_room_door_opened and not self._key_room_door_opened:
            self._key_room_door_opened_step = self._steps
        self._key_room_door_opened = self._key_room_door_opened or key_room_door_opened
        if key_picked_up and not self._key_picked_up:
            self._key_picked_up_step = self._steps
        self._key_picked_up = self._key_picked_up or key_picked_up
        self._key_dropped = self._key_dropped or key_dropped
        if target_door_opened and not self._target_door_opened:
            self._target_door_opened_step = self._steps
        self._target_door_opened = self._target_door_opened or target_door_opened
        self._update_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != (goal_in_front_before_action and action == 2):
            raise RuntimeError("MiniGrid LockedRoom goal semantics drifted")
        if terminated != success:
            raise RuntimeError("MiniGrid LockedRoom termination semantics drifted")
        expected_reward = 1.0 - 0.9 * self._steps / _MAX_STEPS if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid LockedRoom reward semantics drifted")
        if truncated != (self._steps == _MAX_STEPS):
            raise RuntimeError("MiniGrid LockedRoom horizon semantics drifted")
        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated and not truncated
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
        carried_key_color = (
            _COLORS[facts.carried_key_color] if facts.carried_key_color is not None else "none"
        )
        carried_before = (
            _COLORS[carried_before_action] if carried_before_action is not None else "none"
        )
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(_MAX_STEPS - self._steps, 0),
            "target_color": _COLORS[facts.target_color],
            "key_room_color": _COLORS[facts.key_room_color],
            "target_key_label": f"{_COLORS[facts.target_color]}_key",
            "target_key_visible": facts.target_key_visible,
            "key_found": self._key_found,
            "key_first_seen_step": self._key_first_seen_step,
            "key_picked_up_this_step": key_picked_up,
            "key_picked_up": self._key_picked_up,
            "key_picked_up_step": self._key_picked_up_step,
            "key_dropped_this_step": key_dropped,
            "key_dropped": self._key_dropped,
            "carried_key_color": carried_key_color,
            "carried_key_color_before_action": carried_before,
            "matching_target_key_carried": (facts.carried_key_color == facts.target_color),
            "key_room_door_opened_this_step": key_room_door_opened,
            "key_room_door_opened": self._key_room_door_opened,
            "key_room_door_opened_step": self._key_room_door_opened_step,
            "target_door_visible": facts.target_door_visible,
            "target_door_found": self._target_door_found,
            "target_door_first_seen_step": self._target_door_first_seen_step,
            "target_door_opened_this_step": target_door_opened,
            "target_door_opened": self._target_door_opened,
            "target_door_opened_step": self._target_door_opened_step,
            "visible_door_count": len(facts.visible_doors),
            "unique_door_color_count_opened": len(self._opened_door_colors),
            "door_opened_this_step": door_opened,
            "door_closed_this_step": door_closed,
            "door_open_event_count": self._door_open_events,
            "door_close_event_count": self._door_close_events,
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "goal_visible": facts.goal_visible,
            "goal_found": self._goal_found,
            "goal_first_seen_step": self._goal_first_seen_step,
            "goal_in_front": facts.goal_in_front,
            "goal_in_front_before_action": goal_in_front_before_action,
            "pickup_attempt": pickup_attempt,
            "pickup_attempt_count": self._pickup_attempts,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "drop_attempt": drop_attempt,
            "drop_attempt_count": self._drop_attempts,
            "failed_drop": failed_drop,
            "failed_drop_count": self._failed_drop_attempts,
            "toggle_attempt": toggle_attempt,
            "toggle_effective": door_state_changed,
            "toggle_attempt_count": self._toggle_attempts,
            "failed_toggle": failed_toggle,
            "failed_toggle_count": self._failed_toggle_attempts,
            "task_stage": task_stage,
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": (self._novel_observation_steps / self._steps),
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "success_reward_at_this_step": (1.0 - 0.9 * self._steps / _MAX_STEPS),
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

    def _update_discovery(self, facts: _Facts, *, step: int) -> None:
        if facts.target_key_visible and not self._key_found:
            self._key_first_seen_step = step
        if facts.target_door_visible and not self._target_door_found:
            self._target_door_first_seen_step = step
        if facts.goal_visible and not self._goal_found:
            self._goal_first_seen_step = step
        self._key_found = self._key_found or facts.target_key_visible
        self._target_door_found = self._target_door_found or facts.target_door_visible
        self._goal_found = self._goal_found or facts.goal_visible

    def _task_stage(self, facts: _Facts, *, terminal_reason: str) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if facts.goal_in_front:
            return "enter_goal"
        if self._target_door_opened:
            return "approach_goal"
        if facts.carried_key_color == facts.target_color:
            if self._target_door_found:
                return "unlock_target_door"
            return "find_target_door"
        if self._key_picked_up:
            return "recover_target_key"
        if self._key_found:
            return "pick_up_target_key"
        if self._key_room_door_opened:
            return "search_key_room"
        if (
            facts.front_object[0] == _DOOR
            and facts.front_object[1] == facts.key_room_color
            and facts.front_object[2] == _CLOSED
        ):
            return "open_key_room"
        return "find_key_room"


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid LockedRoom returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid LockedRoom returned invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid LockedRoom returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid LockedRoom returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid LockedRoom returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    target_color, key_room_color = _mission_colors(mission)
    carried_key_color = int(image[3, 6, 1]) if image[3, 6, 0] == _KEY else None
    front_object = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
    visible_doors = tuple(
        sorted(
            {
                (int(color_code), int(state_code))
                for object_code, color_code, state_code in image.reshape(-1, 3)
                if int(object_code) == _DOOR
            }
        )
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
        _Facts(
            target_color=target_color,
            key_room_color=key_room_color,
            target_key_visible=bool(
                numpy.any((image[:, :, 0] == _KEY) & (image[:, :, 1] == target_color))
            ),
            carried_key_color=carried_key_color,
            goal_visible=bool(numpy.any(image[:, :, 0] == _GOAL)),
            goal_in_front=int(image[3, 5, 0]) == _GOAL,
            target_door_visible=bool(
                numpy.any(
                    (image[:, :, 0] == _DOOR)
                    & (image[:, :, 1] == target_color)
                    & (image[:, :, 2] == _LOCKED)
                )
            ),
            visible_doors=visible_doors,
            front_object=front_object,
            front_label=_object_label(*front_object),
        ),
    )


def _mission_colors(mission: str) -> tuple[int, int]:
    prefix = "get the "
    suffix = " key from the "
    if not mission.startswith(prefix) or suffix not in mission:
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    target_color, remainder = mission.removeprefix(prefix).split(
        suffix,
        maxsplit=1,
    )
    room_suffix = " room, "
    if room_suffix not in remainder:
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    key_room_color, instruction = remainder.split(room_suffix, maxsplit=1)
    expected = f"unlock the {target_color} door and go to the goal"
    if (
        target_color not in _COLORS
        or key_room_color not in _COLORS
        or key_room_color == target_color
        or instruction != expected
    ):
        raise RuntimeError("MiniGrid LockedRoom returned invalid mission")
    return _COLORS.index(target_color), _COLORS.index(key_room_color)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid LockedRoom returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid LockedRoom returned non-finite reward")
    return number


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
        raise RuntimeError("MiniGrid LockedRoom public observation is invalid")
    return image.data, direction


__all__ = ["LockedRoomEnvironment"]
