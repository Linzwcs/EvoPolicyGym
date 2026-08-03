"""One fresh ObstructedMaze Environment per Episode."""

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

from .config import ObstructedMazeConfig

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
_BALL = 6
_BOX = 7
_OPEN = 0
_CLOSED = 1
_LOCKED = 2
_BLOCKER_COLOR = 1  # Upstream COLOR_NAMES[1] is green.
_BOX_COLOR = 5  # Upstream COLOR_NAMES[2] is grey; encoding differs.
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
    target: tuple[int, int]
    carried: tuple[int, int] | None
    visible_key_colors: tuple[int, ...]
    visible_locked_door_colors: tuple[int, ...]
    visible_door_count: int
    box_visible: bool
    blocker_visible: bool
    target_visible: bool
    front_object: tuple[int, int, int]
    front_label: str

    @property
    def target_in_front(self) -> bool:
        return self.front_object[:2] == self.target

    @property
    def blocker_in_front(self) -> bool:
        return self.front_object[:2] == (_BALL, _BLOCKER_COLOR)

    @property
    def box_in_front(self) -> bool:
        return self.front_object[:2] == (_BOX, _BOX_COLOR)


class ObstructedMazeEnvironment:
    """Strict seeded adapter around the upstream environment."""

    def __init__(
        self,
        episode: EpisodeSpec,
        *,
        config: ObstructedMazeConfig,
    ) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not ObstructedMazeConfig:
            raise TypeError("config must be ObstructedMazeConfig")
        if episode.scenario is not None:
            raise ValueError("ObstructedMaze configuration belongs in ObstructedMazeConfig")
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._started = False
        self._done = False
        self._closed = False
        self._target: tuple[int, int] | None = None
        self._previous_facts: _Facts | None = None
        self._steps = 0
        self._box_found = False
        self._box_first_seen_step = -1
        self._box_open_events = 0
        self._first_box_opened_step = -1
        self._blocker_found = False
        self._blocker_first_seen_step = -1
        self._blocker_pickup_events = 0
        self._first_blocker_pickup_step = -1
        self._blocker_drop_events = 0
        self._first_blocker_relocated_step = -1
        self._key_found = False
        self._key_first_seen_step = -1
        self._key_pickup_events = 0
        self._first_key_pickup_step = -1
        self._key_drop_events = 0
        self._locked_door_found = False
        self._locked_door_first_seen_step = -1
        self._door_open_events = 0
        self._locked_door_open_events = 0
        self._unlocked_door_open_events = 0
        self._door_close_events = 0
        self._door_crossing_events = 0
        self._first_locked_door_opened_step = -1
        self._target_found = False
        self._target_first_seen_step = -1
        self._pickup_attempts = 0
        self._failed_pickup_attempts = 0
        self._drop_attempts = 0
        self._failed_drop_attempts = 0
        self._toggle_attempts = 0
        self._failed_toggle_attempts = 0
        self._target_pickup_blocked_count = 0
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
            raise RuntimeError("MiniGrid ObstructedMaze returned an unexpected horizon")
        self._target = facts.target
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
            raise RuntimeError("MiniGrid ObstructedMaze observation history is unavailable")
        carried_before_action = previous_facts.carried
        front_before_action = previous_facts.front_object
        front_label_before_action = previous_facts.front_label

        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid ObstructedMaze returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        if facts.target != self._target:
            raise RuntimeError("MiniGrid ObstructedMaze changed its mission")
        self._steps += 1
        signature = _observation_signature(public)
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        pickup_attempt = action == 3
        object_picked_up = bool(
            pickup_attempt and carried_before_action is None and facts.carried is not None
        )
        blocker_picked_up = bool(object_picked_up and facts.carried == (_BALL, _BLOCKER_COLOR))
        key_picked_up = bool(
            object_picked_up and facts.carried is not None and facts.carried[0] == _KEY
        )
        target_picked_up = bool(object_picked_up and facts.carried == facts.target)
        failed_pickup = pickup_attempt and not object_picked_up
        target_pickup_blocked = bool(
            pickup_attempt and previous_facts.target_in_front and carried_before_action is not None
        )
        self._pickup_attempts += int(pickup_attempt)
        self._failed_pickup_attempts += int(failed_pickup)
        self._target_pickup_blocked_count += int(target_pickup_blocked)
        if blocker_picked_up:
            self._blocker_pickup_events += 1
            if self._first_blocker_pickup_step < 0:
                self._first_blocker_pickup_step = self._steps
        if key_picked_up:
            self._key_pickup_events += 1
            if self._first_key_pickup_step < 0:
                self._first_key_pickup_step = self._steps

        drop_attempt = action == 4
        object_dropped = bool(
            drop_attempt and carried_before_action is not None and facts.carried is None
        )
        blocker_dropped = object_dropped and carried_before_action == (_BALL, _BLOCKER_COLOR)
        key_dropped = bool(
            object_dropped
            and carried_before_action is not None
            and carried_before_action[0] == _KEY
        )
        failed_drop = drop_attempt and not object_dropped
        self._drop_attempts += int(drop_attempt)
        self._failed_drop_attempts += int(failed_drop)
        if blocker_dropped:
            self._blocker_drop_events += 1
            if self._first_blocker_relocated_step < 0:
                self._first_blocker_relocated_step = self._steps
        self._key_drop_events += int(key_dropped)

        toggle_attempt = action == 5
        box_opened = bool(
            toggle_attempt and front_before_action[0] == _BOX and facts.front_object[0] == _KEY
        )
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
        locked_door_opened = door_opened and front_before_action[2] == _LOCKED
        unlocked_door_opened = door_opened and front_before_action[2] == _CLOSED
        door_closed = bool(
            door_state_changed
            and front_before_action[2] == _OPEN
            and facts.front_object[2] == _CLOSED
        )
        toggle_effective = box_opened or door_state_changed
        failed_toggle = toggle_attempt and not toggle_effective
        self._toggle_attempts += int(toggle_attempt)
        self._failed_toggle_attempts += int(failed_toggle)
        if box_opened:
            self._box_open_events += 1
            if self._first_box_opened_step < 0:
                self._first_box_opened_step = self._steps
        if locked_door_opened:
            if carried_before_action != (_KEY, front_before_action[1]):
                raise RuntimeError("MiniGrid ObstructedMaze unlock semantics drifted")
            self._locked_door_open_events += 1
            if self._first_locked_door_opened_step < 0:
                self._first_locked_door_opened_step = self._steps
        self._unlocked_door_open_events += int(unlocked_door_opened)
        self._door_open_events += int(door_opened)
        self._door_close_events += int(door_closed)

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

        self._update_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != target_picked_up:
            raise RuntimeError("MiniGrid ObstructedMaze target pickup semantics drifted")
        if terminated != success:
            raise RuntimeError("MiniGrid ObstructedMaze termination semantics drifted")
        horizon = self._config.max_episode_steps
        expected_reward = 1.0 - 0.9 * self._steps / horizon if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid ObstructedMaze reward semantics drifted")
        if truncated != (self._steps == horizon):
            raise RuntimeError("MiniGrid ObstructedMaze horizon semantics drifted")
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
        task_stage = self._task_stage(
            facts,
            terminal_reason=terminal_reason,
            blocker_dropped_this_step=blocker_dropped,
            key_dropped_this_step=key_dropped,
        )
        self._done = terminated or truncated
        self._previous_facts = facts

        carried_label = _carried_label(facts.carried)
        carried_before_label = _carried_label(carried_before_action)
        visible_key_labels: list[PolicyValue] = [
            f"{_COLORS[color]}_key" for color in facts.visible_key_colors
        ]
        visible_locked_door_labels: list[PolicyValue] = [
            f"{_COLORS[color]}_locked_door" for color in facts.visible_locked_door_colors
        ]
        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(horizon - self._steps, 0),
            "target_label": f"{_COLORS[facts.target[1]]}_ball",
            "blocker_label": f"{_COLORS[_BLOCKER_COLOR]}_ball",
            "key_box_label": f"{_COLORS[_BOX_COLOR]}_box",
            "required_locked_door_count": self._config.locked_doors,
            "required_unlocked_door_count": self._config.unlocked_doors,
            "box_visible": facts.box_visible,
            "box_found": self._box_found,
            "box_first_seen_step": self._box_first_seen_step,
            "box_opened_this_step": box_opened,
            "box_opened": self._box_open_events > 0,
            "box_open_event_count": self._box_open_events,
            "first_box_opened_step": self._first_box_opened_step,
            "blocker_visible": facts.blocker_visible,
            "blocker_found": self._blocker_found,
            "blocker_first_seen_step": self._blocker_first_seen_step,
            "blocker_picked_up_this_step": blocker_picked_up,
            "blocker_picked_up": self._blocker_pickup_events > 0,
            "blocker_pickup_event_count": self._blocker_pickup_events,
            "first_blocker_pickup_step": self._first_blocker_pickup_step,
            "blocker_dropped_this_step": blocker_dropped,
            "blocker_relocated": self._blocker_drop_events > 0,
            "blocker_drop_event_count": self._blocker_drop_events,
            "first_blocker_relocated_step": self._first_blocker_relocated_step,
            "visible_key_count": len(facts.visible_key_colors),
            "visible_key_labels": visible_key_labels,
            "key_found": self._key_found,
            "key_first_seen_step": self._key_first_seen_step,
            "key_picked_up_this_step": key_picked_up,
            "key_picked_up": self._key_pickup_events > 0,
            "key_pickup_event_count": self._key_pickup_events,
            "first_key_pickup_step": self._first_key_pickup_step,
            "key_dropped_this_step": key_dropped,
            "key_drop_event_count": self._key_drop_events,
            "visible_door_count": facts.visible_door_count,
            "visible_locked_door_count": len(facts.visible_locked_door_colors),
            "visible_locked_door_labels": visible_locked_door_labels,
            "locked_door_found": self._locked_door_found,
            "locked_door_first_seen_step": self._locked_door_first_seen_step,
            "door_opened_this_step": door_opened,
            "locked_door_opened_this_step": locked_door_opened,
            "locked_door_opened": self._locked_door_open_events > 0,
            "unlocked_door_opened_this_step": unlocked_door_opened,
            "door_closed_this_step": door_closed,
            "door_open_event_count": self._door_open_events,
            "locked_door_open_event_count": self._locked_door_open_events,
            "unlocked_door_open_event_count": self._unlocked_door_open_events,
            "door_close_event_count": self._door_close_events,
            "first_locked_door_opened_step": self._first_locked_door_opened_step,
            "door_crossed_this_step": door_crossed,
            "door_crossing_event_count": self._door_crossing_events,
            "target_visible": facts.target_visible,
            "target_found": self._target_found,
            "target_first_seen_step": self._target_first_seen_step,
            "target_in_front": facts.target_in_front,
            "target_in_front_before_action": previous_facts.target_in_front,
            "target_picked_up_this_step": target_picked_up,
            "carried_object": carried_label,
            "carried_object_before_action": carried_before_label,
            "matching_key_for_front_locked_door_carried": bool(
                facts.front_object[0] == _DOOR
                and facts.front_object[2] == _LOCKED
                and facts.carried == (_KEY, facts.front_object[1])
            ),
            "front_object": facts.front_label,
            "front_object_before_action": front_label_before_action,
            "pickup_attempt": pickup_attempt,
            "pickup_attempt_count": self._pickup_attempts,
            "failed_pickup": failed_pickup,
            "failed_pickup_count": self._failed_pickup_attempts,
            "target_pickup_blocked_by_carried_object": target_pickup_blocked,
            "target_pickup_blocked_count": self._target_pickup_blocked_count,
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

    def _update_discovery(self, facts: _Facts, *, step: int) -> None:
        if facts.box_visible and not self._box_found:
            self._box_first_seen_step = step
        if facts.blocker_visible and not self._blocker_found:
            self._blocker_first_seen_step = step
        if facts.visible_key_colors and not self._key_found:
            self._key_first_seen_step = step
        if facts.visible_locked_door_colors and not self._locked_door_found:
            self._locked_door_first_seen_step = step
        if facts.target_visible and not self._target_found:
            self._target_first_seen_step = step
        self._box_found = self._box_found or facts.box_visible
        self._blocker_found = self._blocker_found or facts.blocker_visible
        self._key_found = self._key_found or bool(facts.visible_key_colors)
        self._locked_door_found = self._locked_door_found or bool(facts.visible_locked_door_colors)
        self._target_found = self._target_found or facts.target_visible

    def _task_stage(
        self,
        facts: _Facts,
        *,
        terminal_reason: str,
        blocker_dropped_this_step: bool,
        key_dropped_this_step: bool,
    ) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if facts.target_in_front:
            return "pick_up_target" if facts.carried is None else "free_hands_for_target"
        if blocker_dropped_this_step:
            return "continue_after_relocating_blocker"
        if key_dropped_this_step:
            return "continue_after_releasing_key"
        if self._target_found:
            return "approach_target" if facts.carried is None else "free_hands_for_target"
        if facts.carried == (_BALL, _BLOCKER_COLOR):
            return "relocate_blocker"
        if facts.blocker_in_front:
            return "pick_up_blocker" if facts.carried is None else "free_hands_for_blocker"
        if facts.box_in_front:
            return "open_key_box"
        if facts.front_object[0] == _KEY:
            return "pick_up_key" if facts.carried is None else "free_hands_for_key"
        if facts.front_object[0] == _DOOR and facts.front_object[2] == _LOCKED:
            if facts.carried == (_KEY, facts.front_object[1]):
                return "unlock_door"
            return "find_matching_key"
        if facts.carried is not None and facts.carried[0] == _KEY:
            return "find_matching_locked_door"
        if self._locked_door_open_events:
            return "search_unlocked_rooms"
        return "explore_maze"


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {"image", "direction", "mission"}:
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid ObstructedMaze returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str:
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid mission")
    target = _target(mission)
    carried_code = int(image[3, 6, 0])
    carried = (carried_code, int(image[3, 6, 1])) if carried_code in {_KEY, _BALL, _BOX} else None
    front_object = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
    objects = image[:, :, 0]
    colors = image[:, :, 1]
    states = image[:, :, 2]
    visible_key_colors = tuple(sorted({int(color) for color in colors[objects == _KEY]}))
    visible_locked_door_colors = tuple(
        sorted({int(color) for color in colors[(objects == _DOOR) & (states == _LOCKED)]})
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
            target=target,
            carried=carried,
            visible_key_colors=visible_key_colors,
            visible_locked_door_colors=visible_locked_door_colors,
            visible_door_count=int(numpy.count_nonzero(objects == _DOOR)),
            box_visible=bool(numpy.any((objects == _BOX) & (colors == _BOX_COLOR))),
            blocker_visible=bool(numpy.any((objects == _BALL) & (colors == _BLOCKER_COLOR))),
            target_visible=bool(numpy.any((objects == target[0]) & (colors == target[1]))),
            front_object=front_object,
            front_label=_object_label(*front_object),
        ),
    )


def _target(mission: str) -> tuple[int, int]:
    prefix = "pick up the "
    if not mission.startswith(prefix):
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid mission")
    parts = mission.removeprefix(prefix).split(" ")
    if len(parts) != 2 or parts[0] not in _COLORS or parts[1] != "ball":
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid mission")
    target = (_BALL, _COLORS.index(parts[0]))
    if target != (_BALL, 2):
        raise RuntimeError("MiniGrid ObstructedMaze returned an unexpected target")
    return target


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid ObstructedMaze returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid ObstructedMaze returned non-finite reward")
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
        raise RuntimeError("MiniGrid ObstructedMaze public observation is invalid")
    return image.data, direction


__all__ = ["ObstructedMazeEnvironment"]
