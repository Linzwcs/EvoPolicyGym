"""One fresh MiniGrid Playground room-coverage Environment per Episode."""

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

_ENVIRONMENT_ID = "MiniGrid-Playground-v0"
_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MAX_EPISODE_STEPS = 1000
_ROOMS = 9
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
_BALL = 6
_BOX = 7
_OPEN = 0
_CLOSED = 1
_PORTABLE = frozenset({_KEY, _BALL, _BOX})
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
    carried: tuple[int, int] | None
    visible_door_count: int
    visible_closed_door_count: int
    visible_open_door_count: int
    visible_portable_object_count: int
    front_object: tuple[int, int, int]
    front_label: str


class PlaygroundEnvironment:
    """Strict seeded adapter that rewards first entry into each room."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Playground has no Episode scenario overrides")
        self._seed = episode.environment_seed
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(_ENVIRONMENT_ID, max_steps=_MAX_EPISODE_STEPS),
        )
        self._visited_rooms: set[tuple[int, int]] = set()
        self._room_first_entry_steps = [-1] * _ROOMS
        self._started = False
        self._done = False
        self._closed = False
        self._previous_facts: _Facts | None = None
        self._steps = 0
        self._last_new_room_step = 0
        self._door_found = False
        self._door_first_seen_step = -1
        self._door_open_events = 0
        self._door_close_events = 0
        self._door_crossing_events = 0
        self._pickup_attempts = 0
        self._pickup_events = 0
        self._failed_pickup_attempts = 0
        self._drop_attempts = 0
        self._drop_events = 0
        self._failed_drop_attempts = 0
        self._toggle_attempts = 0
        self._failed_toggle_attempts = 0
        self._box_destroy_events = 0
        self._blocked_forward_count = 0
        self._done_action_count = 0
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
        if type(horizon) is not int or horizon != _MAX_EPISODE_STEPS:
            raise RuntimeError("MiniGrid Playground returned an unexpected horizon")
        self._visited_rooms.add(self._room())
        self._room_first_entry_steps[0] = 0
        self._update_door_discovery(facts, step=0)
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
            raise RuntimeError("MiniGrid Playground observation history is unavailable")
        carried_before_action = previous_facts.carried
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label

        observation, upstream_reward, upstream_terminated, truncated, _ = self._environment.step(
            action
        )
        if type(upstream_terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid Playground returned invalid flags")
        if upstream_terminated:
            raise RuntimeError("MiniGrid Playground unexpectedly terminated naturally")
        if _number(upstream_reward) != 0.0:
            raise RuntimeError("MiniGrid Playground unexpectedly returned a reward")
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        room = self._room()
        new_room = room not in self._visited_rooms
        if new_room:
            self._visited_rooms.add(room)
            rooms_visited = len(self._visited_rooms)
            self._room_first_entry_steps[rooms_visited - 1] = self._steps
            self._last_new_room_step = self._steps
        rooms_visited = len(self._visited_rooms)
        rooms_remaining = _ROOMS - rooms_visited

        pickup_attempt = action == 3
        object_picked_up = bool(
            pickup_attempt and carried_before_action is None and facts.carried is not None
        )
        failed_pickup = pickup_attempt and not object_picked_up
        self._pickup_attempts += int(pickup_attempt)
        self._pickup_events += int(object_picked_up)
        self._failed_pickup_attempts += int(failed_pickup)

        drop_attempt = action == 4
        object_dropped = bool(
            drop_attempt and carried_before_action is not None and facts.carried is None
        )
        failed_drop = drop_attempt and not object_dropped
        self._drop_attempts += int(drop_attempt)
        self._drop_events += int(object_dropped)
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
            and front_before_action[2] == _CLOSED
            and facts.front_object[2] == _OPEN
        )
        door_closed = bool(
            door_state_changed
            and front_before_action[2] == _OPEN
            and facts.front_object[2] == _CLOSED
        )
        box_destroyed = bool(
            toggle_attempt and front_before_action[0] == _BOX and facts.front_object[0] != _BOX
        )
        toggle_effective = door_state_changed or box_destroyed
        failed_toggle = toggle_attempt and not toggle_effective
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        self._door_open_events += int(door_opened)
        self._door_close_events += int(door_closed)
        self._box_destroy_events += int(box_destroyed)

        forward_attempt = action == 2
        blocked_forward = bool(forward_attempt and signature == previous_signature)
        door_crossed = bool(
            forward_attempt
            and front_before_action[0] == _DOOR
            and front_before_action[2] == _OPEN
            and not blocked_forward
        )
        self._blocked_forward_count += int(blocked_forward)
        self._door_crossing_events += int(door_crossed)
        done_action = action == 6
        self._done_action_count += int(done_action)
        self._update_door_discovery(facts, step=self._steps)

        success = rooms_visited == _ROOMS
        terminated = success
        if truncated != (self._steps == _MAX_EPISODE_STEPS):
            raise RuntimeError("MiniGrid Playground horizon semantics drifted")
        reward = 1.0 if new_room else 0.0
        coverage = rooms_visited / _ROOMS
        self._cumulative_return += reward
        expected_return = float(rooms_visited - 1)
        if self._cumulative_return != expected_return:
            raise RuntimeError("MiniGrid Playground coverage reward drifted")
        ineffective_action = bool(
            signature == previous_signature and reward == 0.0 and not terminated
        )
        self._ineffective_actions += int(ineffective_action)
        terminal_reason = "none"
        if success and truncated:
            terminal_reason = "success_and_time_limit"
        elif success:
            terminal_reason = "success"
        elif truncated:
            terminal_reason = "time_limit"
        task_stage = self._task_stage(
            facts,
            new_room=new_room,
            terminal_reason=terminal_reason,
        )
        self._done = terminated or truncated
        self._previous_facts = facts

        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(_MAX_EPISODE_STEPS - self._steps, 0),
            "rooms_visited": rooms_visited,
            "rooms_remaining": rooms_remaining,
            "room_coverage": coverage,
            "coverage_gain": 1.0 / _ROOMS if new_room else 0.0,
            "new_room": new_room,
            "new_room_entry_count": rooms_visited - 1,
            "last_new_room_step": self._last_new_room_step,
            "steps_since_new_room": self._steps - self._last_new_room_step,
            "visible_door_count": facts.visible_door_count,
            "visible_closed_door_count": facts.visible_closed_door_count,
            "visible_open_door_count": facts.visible_open_door_count,
            "door_found": self._door_found,
            "door_first_seen_step": self._door_first_seen_step,
            "door_opened_this_step": door_opened,
            "door_closed_this_step": door_closed,
            "door_open_event_count": self._door_open_events,
            "door_close_event_count": self._door_close_events,
            "door_crossed_this_step": door_crossed,
            "door_crossing_event_count": self._door_crossing_events,
            "visible_portable_object_count": facts.visible_portable_object_count,
            "carried_object": _carried_label(facts.carried),
            "carried_object_before_action": _carried_label(carried_before_action),
            "object_picked_up_this_step": object_picked_up,
            "pickup_event_count": self._pickup_events,
            "object_dropped_this_step": object_dropped,
            "drop_event_count": self._drop_events,
            "box_destroyed_this_step": box_destroyed,
            "box_destroy_event_count": self._box_destroy_events,
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "pickup_attempt": pickup_attempt,
            "pickup_attempt_count": self._pickup_attempts,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "drop_attempt": drop_attempt,
            "drop_attempt_count": self._drop_attempts,
            "failed_drop": failed_drop,
            "failed_drop_count": self._failed_drop_attempts,
            "toggle_attempt": toggle_attempt,
            "toggle_effective": toggle_effective,
            "toggle_attempt_count": self._toggle_attempts,
            "failed_toggle": failed_toggle,
            "failed_toggle_count": self._failed_toggle_attempts,
            "blocked_forward": blocked_forward,
            "blocked_forward_count": self._blocked_forward_count,
            "done_action": done_action,
            "done_action_count": self._done_action_count,
            "task_stage": task_stage,
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_observation_signatures),
            "observation_novelty_step_fraction": self._novel_observation_steps / self._steps,
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "cumulative_return": self._cumulative_return,
            "success": success,
            "terminal_reason": terminal_reason,
        }
        for room_count in range(2, _ROOMS + 1):
            metrics[f"room_{room_count}_first_entry_step"] = self._room_first_entry_steps[
                room_count - 1
            ]
        for name, count in zip(_ACTION_NAMES, self._action_counts, strict=True):
            metrics[f"{name}_count"] = count
        return Step(
            observation=public,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _room(self) -> tuple[int, int]:
        value = self._environment.get_wrapper_attr("agent_pos")
        if not isinstance(value, (tuple, list, numpy.ndarray)) or len(value) != 2:
            raise RuntimeError("MiniGrid Playground returned invalid position")
        try:
            x = operator.index(cast(SupportsIndex, value[0]))
            y = operator.index(cast(SupportsIndex, value[1]))
        except TypeError as error:
            raise RuntimeError("MiniGrid Playground returned invalid position") from error
        if not 1 <= x <= 17 or not 1 <= y <= 17:
            raise RuntimeError("MiniGrid Playground returned invalid position")
        return min((x - 1) // 6, 2), min((y - 1) // 6, 2)

    def _update_door_discovery(self, facts: _Facts, *, step: int) -> None:
        if facts.visible_door_count > 0 and not self._door_found:
            self._door_first_seen_step = step
        self._door_found = self._door_found or facts.visible_door_count > 0

    def _task_stage(
        self,
        facts: _Facts,
        *,
        new_room: bool,
        terminal_reason: str,
    ) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if new_room:
            return "explore_new_room"
        if facts.front_object[0] == _DOOR and facts.front_object[2] == _CLOSED:
            return "open_door"
        if facts.carried is not None:
            return "relocate_carried_object"
        if self._steps - self._last_new_room_step >= 100:
            return "coverage_stalled"
        return "find_unvisited_room"


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {"image", "direction", "mission"}:
        raise RuntimeError("MiniGrid Playground returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Playground returned invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid Playground returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid Playground returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Playground returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != "":
        raise RuntimeError("MiniGrid Playground returned invalid mission")
    objects = image[:, :, 0]
    states = image[:, :, 2]
    door_mask = objects == _DOOR
    front_object = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
    carried_code = int(image[3, 6, 0])
    carried = (carried_code, int(image[3, 6, 1])) if carried_code in _PORTABLE else None
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
            carried=carried,
            visible_door_count=int(numpy.count_nonzero(door_mask)),
            visible_closed_door_count=int(numpy.count_nonzero(door_mask & (states == _CLOSED))),
            visible_open_door_count=int(numpy.count_nonzero(door_mask & (states == _OPEN))),
            visible_portable_object_count=int(
                numpy.count_nonzero(numpy.isin(objects, tuple(_PORTABLE)))
            ),
            front_object=front_object,
            front_label=_object_label(*front_object),
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid Playground returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid Playground returned non-finite reward")
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


def _carried_label(carried: tuple[int, int] | None) -> str:
    if carried is None:
        return "none"
    return _object_label(carried[0], carried[1], _OPEN)


def _observation_signature(observation: dict[str, PolicyValue]) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid Playground public observation is invalid")
    return image.data, direction


__all__ = ["PlaygroundEnvironment"]
