"""One fresh MiniGrid PutNear Environment per Episode."""

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

from .config import PutNearConfig

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
_OBJECT_CODES = {"key": 5, "ball": 6, "box": 7}
_PORTABLE = frozenset(_OBJECT_CODES.values())
_BOX = 7
_EMPTY = 1
_OPEN = 0
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
    move_object: tuple[int, int]
    target_object: tuple[int, int]
    move_visible: bool
    target_visible: bool
    carried_object: tuple[int, int] | None
    visible_portable_object_count: int
    front_object: tuple[int, int, int]
    front_label: str
    front_cell_near_target: bool

    @property
    def move_in_front(self) -> bool:
        return self.front_object[:2] == self.move_object

    @property
    def target_in_front(self) -> bool:
        return self.front_object[:2] == self.target_object

    @property
    def front_cell_empty(self) -> bool:
        return self.front_object[0] == _EMPTY


class PutNearEnvironment:
    """The seeded strict adapter around configured MiniGrid PutNear."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: PutNearConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not PutNearConfig:
            raise TypeError("config must be PutNearConfig")
        if episode.scenario is not None:
            raise ValueError(
                "PutNear configuration belongs in PutNearConfig, not EpisodeSpec.scenario"
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
        self._mission_objects: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._previous_facts: _ObservationFacts | None = None
        self._steps = 0
        self._move_found = False
        self._move_first_seen_step = -1
        self._target_found = False
        self._target_first_seen_step = -1
        self._correct_pickup_step = -1
        self._move_object_destroyed = False
        self._target_object_destroyed = False
        self._mission_object_destroyed_step = -1
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
        if type(horizon) is not int or horizon != self._config.max_episode_steps:
            raise RuntimeError("MiniGrid PutNear returned an unexpected horizon")
        self._mission_objects = (facts.move_object, facts.target_object)
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
            raise RuntimeError("MiniGrid PutNear observation history is unavailable")
        carried_before_action = previous_facts.carried_object
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label
        valid_success_drop_before_action = bool(
            carried_before_action == previous_facts.move_object
            and previous_facts.front_cell_empty
            and previous_facts.front_cell_near_target
        )

        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid PutNear returned invalid termination flags")
        number = _number(reward, name="reward")
        public, facts = _observation(observation)
        if (facts.move_object, facts.target_object) != self._mission_objects:
            raise RuntimeError("MiniGrid PutNear changed mission objects during an Episode")
        self._steps += 1
        signature = _observation_signature(public)
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        pickup_attempt = action == 3
        object_picked_up = bool(
            pickup_attempt and carried_before_action is None and facts.carried_object is not None
        )
        correct_object_picked_up = bool(
            object_picked_up and facts.carried_object == facts.move_object
        )
        wrong_object_picked_up = bool(
            object_picked_up and facts.carried_object != facts.move_object
        )
        failed_pickup = pickup_attempt and not object_picked_up
        self._pickup_attempts += int(pickup_attempt)
        self._pickup_events += int(object_picked_up)
        self._failed_pickup_attempts += int(failed_pickup)
        if correct_object_picked_up and self._correct_pickup_step < 0:
            self._correct_pickup_step = self._steps

        drop_attempt = action == 4
        terminal_drop_attempt = drop_attempt and carried_before_action is not None
        object_dropped = bool(terminal_drop_attempt and facts.carried_object is None)
        failed_drop = drop_attempt and not object_dropped
        self._drop_attempts += int(drop_attempt)
        self._drop_events += int(object_dropped)
        self._failed_drop_attempts += int(failed_drop)

        toggle_attempt = action == 5
        box_destroyed = bool(
            toggle_attempt and front_before_action[0] == _BOX and facts.front_object[0] != _BOX
        )
        move_object_destroyed = box_destroyed and front_before_action[:2] == facts.move_object
        target_object_destroyed = box_destroyed and front_before_action[:2] == facts.target_object
        toggle_effective = box_destroyed
        failed_toggle = toggle_attempt and not toggle_effective
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        self._box_destroy_events += int(box_destroyed)
        if (move_object_destroyed or target_object_destroyed) and not (
            self._move_object_destroyed or self._target_object_destroyed
        ):
            self._mission_object_destroyed_step = self._steps
        self._move_object_destroyed = self._move_object_destroyed or move_object_destroyed
        self._target_object_destroyed = self._target_object_destroyed or target_object_destroyed

        forward_attempt = action == 2
        blocked_forward = bool(forward_attempt and signature == previous_signature)
        self._blocked_forward_count += int(blocked_forward)
        done_action = action == 6
        self._done_action_count += int(done_action)
        self._update_discovery(facts, step=self._steps)

        success = bool(terminated and number > 0.0)
        misplaced_drop = bool(terminal_drop_attempt and object_dropped and not success)
        blocked_terminal_drop = bool(terminal_drop_attempt and not object_dropped and not success)
        expected_termination = wrong_object_picked_up or terminal_drop_attempt
        if terminated != expected_termination:
            raise RuntimeError("MiniGrid PutNear termination semantics drifted")
        if success != (terminal_drop_attempt and valid_success_drop_before_action):
            raise RuntimeError("MiniGrid PutNear placement semantics drifted")
        horizon = self._config.max_episode_steps
        expected_reward = 1.0 - 0.9 * self._steps / horizon if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid PutNear reward semantics drifted")
        if truncated != (self._steps == horizon):
            raise RuntimeError("MiniGrid PutNear horizon semantics drifted")
        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated
        )
        self._ineffective_actions += int(ineffective_action)
        self._cumulative_return += number
        terminal_reason = _terminal_reason(
            success=success,
            wrong_object_picked_up=wrong_object_picked_up,
            misplaced_drop=misplaced_drop,
            blocked_terminal_drop=blocked_terminal_drop,
            truncated=truncated,
        )
        task_stage = self._task_stage(facts, terminal_reason=terminal_reason)
        self._done = terminated or truncated
        self._previous_facts = facts

        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(horizon - self._steps, 0),
            "move_object_label": _object_name(facts.move_object),
            "target_object_label": _object_name(facts.target_object),
            "move_object_visible": facts.move_visible,
            "move_object_found": self._move_found,
            "move_object_first_seen_step": self._move_first_seen_step,
            "move_object_in_front": facts.move_in_front,
            "move_object_in_front_before_action": previous_facts.move_in_front,
            "target_object_visible": facts.target_visible,
            "target_object_found": self._target_found,
            "target_object_first_seen_step": self._target_first_seen_step,
            "target_object_in_front": facts.target_in_front,
            "visible_portable_object_count": facts.visible_portable_object_count,
            "carried_object": _carried_label(facts.carried_object),
            "carried_object_before_action": _carried_label(carried_before_action),
            "carrying_move_object": facts.carried_object == facts.move_object,
            "correct_object_picked_up_this_step": correct_object_picked_up,
            "correct_object_picked_up": self._correct_pickup_step >= 0,
            "correct_object_pickup_step": self._correct_pickup_step,
            "wrong_object_picked_up": wrong_object_picked_up,
            "front_cell_empty": facts.front_cell_empty,
            "front_cell_near_target": facts.front_cell_near_target,
            "valid_success_drop_available": bool(
                facts.carried_object == facts.move_object
                and facts.front_cell_empty
                and facts.front_cell_near_target
            ),
            "valid_success_drop_before_action": valid_success_drop_before_action,
            "terminal_drop_attempt": terminal_drop_attempt,
            "object_dropped_this_step": object_dropped,
            "misplaced_drop": misplaced_drop,
            "blocked_terminal_drop": blocked_terminal_drop,
            "box_destroyed_this_step": box_destroyed,
            "box_destroy_event_count": self._box_destroy_events,
            "move_object_destroyed_this_step": move_object_destroyed,
            "move_object_destroyed": self._move_object_destroyed,
            "target_object_destroyed_this_step": target_object_destroyed,
            "target_object_destroyed": self._target_object_destroyed,
            "mission_object_destroyed": (
                self._move_object_destroyed or self._target_object_destroyed
            ),
            "mission_object_destroyed_step": self._mission_object_destroyed_step,
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "pickup_attempt": pickup_attempt,
            "pickup_attempt_count": self._pickup_attempts,
            "pickup_event_count": self._pickup_events,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "drop_attempt": drop_attempt,
            "drop_attempt_count": self._drop_attempts,
            "drop_event_count": self._drop_events,
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

    def _update_discovery(
        self,
        facts: _ObservationFacts,
        *,
        step: int,
    ) -> None:
        if facts.move_visible and not self._move_found:
            self._move_first_seen_step = step
        if facts.target_visible and not self._target_found:
            self._target_first_seen_step = step
        self._move_found = self._move_found or facts.move_visible
        self._target_found = self._target_found or facts.target_visible

    def _task_stage(
        self,
        facts: _ObservationFacts,
        *,
        terminal_reason: str,
    ) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if self._move_object_destroyed or self._target_object_destroyed:
            return "mission_object_destroyed"
        if facts.carried_object == facts.move_object:
            if facts.front_cell_empty and facts.front_cell_near_target:
                return "drop_near_target"
            if self._target_found:
                return "find_valid_drop_cell_near_target"
            return "find_target_object"
        if facts.move_in_front:
            return "pick_up_move_object"
        if self._move_found:
            return "approach_move_object"
        return "find_move_object"


def _observation(
    value: object,
) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {"image", "direction", "mission"}:
        raise RuntimeError("MiniGrid PutNear returned an invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid PutNear returned an invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid PutNear returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid PutNear returned an invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid PutNear returned an invalid direction")
    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid PutNear returned an invalid mission")
    move_object, target_object = _mission_objects(mission)
    carried_type = int(image[3, 6, 0])
    carried_object = (carried_type, int(image[3, 6, 1])) if carried_type in _PORTABLE else None
    front_object = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
    target_positions = numpy.argwhere(
        (image[:, :, 0] == target_object[0]) & (image[:, :, 1] == target_object[1])
    )
    front_cell_near_target = any(
        abs(int(position[0]) - 3) <= 1 and abs(int(position[1]) - 5) <= 1
        for position in target_positions
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
            move_object=move_object,
            target_object=target_object,
            move_visible=_visible(image, move_object),
            target_visible=_visible(image, target_object),
            carried_object=carried_object,
            visible_portable_object_count=int(
                numpy.count_nonzero(numpy.isin(image[:, :, 0], tuple(_PORTABLE)))
            ),
            front_object=front_object,
            front_label=_object_label(*front_object),
            front_cell_near_target=front_cell_near_target,
        ),
    )


def _mission_objects(
    mission: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    prefix = "put the "
    separator = " near the "
    if not mission.startswith(prefix) or mission.count(separator) != 1:
        raise RuntimeError("MiniGrid PutNear returned an invalid mission")
    move_text, target_text = mission.removeprefix(prefix).split(separator)
    move_object = _named_object(move_text)
    target_object = _named_object(target_text)
    if move_object == target_object:
        raise RuntimeError("MiniGrid PutNear returned identical mission objects")
    return move_object, target_object


def _named_object(text: str) -> tuple[int, int]:
    parts = text.split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] not in _OBJECT_CODES:
        raise RuntimeError("MiniGrid PutNear returned an invalid mission")
    return _OBJECT_CODES[parts[1]], _COLORS.index(parts[0])


def _visible(
    image: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.uint8]],
    target: tuple[int, int],
) -> bool:
    return bool(numpy.any((image[:, :, 0] == target[0]) & (image[:, :, 1] == target[1])))


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError(f"MiniGrid PutNear returned an invalid {name}")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"MiniGrid PutNear returned a non-finite {name}")
    return number


def _terminal_reason(
    *,
    success: bool,
    wrong_object_picked_up: bool,
    misplaced_drop: bool,
    blocked_terminal_drop: bool,
    truncated: bool,
) -> str:
    if success and truncated:
        return "success_and_time_limit"
    if success:
        return "success"
    reason = "none"
    if wrong_object_picked_up:
        reason = "wrong_object_pickup"
    elif misplaced_drop:
        reason = "misplaced_drop"
    elif blocked_terminal_drop:
        reason = "blocked_drop"
    if truncated:
        return "time_limit" if reason == "none" else f"{reason}_and_time_limit"
    return reason


def _object_label(object_code: int, color_code: int, state_code: int) -> str:
    if not 0 <= object_code < len(_OBJECTS):
        return "unknown"
    if object_code == 0:
        return "unseen"
    if object_code == 1:
        return "empty"
    if not 0 <= color_code < len(_COLORS):
        return "unknown"
    return f"{_COLORS[color_code]}_{_OBJECTS[object_code]}"


def _object_name(value: tuple[int, int]) -> str:
    return _object_label(value[0], value[1], _OPEN)


def _carried_label(value: tuple[int, int] | None) -> str:
    return "none" if value is None else _object_name(value)


def _observation_signature(
    observation: dict[str, PolicyValue],
) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid PutNear public observation is invalid")
    return image.data, direction


__all__ = ["PutNearEnvironment"]
