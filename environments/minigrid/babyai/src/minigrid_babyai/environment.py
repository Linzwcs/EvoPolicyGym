"""One fresh strict BabyAI Environment per Episode."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import minigrid  # noqa: F401  # Import registers BabyAI environments.
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from minigrid.envs.babyai.core import verifier

from .config import BabyAIConfig

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_ACTION_NAMES = (
    "turn_left",
    "turn_right",
    "move_forward",
    "pick_up",
    "drop",
    "toggle",
    "done",
)
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
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey")
_STATES = ("open", "closed", "locked")
_NO_OBJECT = "none"


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    direction: int
    mission: str
    front_cell: tuple[int, int, int]
    front_label: str
    visible_object_labels: tuple[str, ...]


class BabyAIEnvironment:
    """Strict seeded adapter with instruction-action diagnostics."""

    def __init__(self, episode: EpisodeSpec, *, config: BabyAIConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not BabyAIConfig:
            raise TypeError("config must be BabyAIConfig")
        if episode.scenario is not None:
            raise ValueError("BabyAI configuration belongs in BabyAIConfig")
        if verifier.use_done_actions:
            raise RuntimeError("BABYAI_DONE_ACTIONS must be unset for stable Benchmark semantics")
        self._seed = episode.environment_seed
        self._config = config
        self._environment = cast(
            gymnasium.Env[object, int],
            gymnasium.make(config.environment_id),
        )
        self._mission: str | None = None
        self._horizon = 0
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._previous_facts: _ObservationFacts | None = None
        self._previous_signature: tuple[bytes, int] | None = None
        self._seen_signatures: set[tuple[bytes, int]] = set()
        self._discovered_object_labels: set[str] = set()
        self._novel_observation_steps = 0
        self._ineffective_actions = 0
        self._blocked_forward_count = 0
        self._pickup_attempts = 0
        self._pickup_events = 0
        self._failed_pickups = 0
        self._drop_attempts = 0
        self._drop_events = 0
        self._failed_drops = 0
        self._toggle_attempts = 0
        self._door_open_events = 0
        self._door_close_events = 0
        self._box_open_events = 0
        self._failed_toggles = 0
        self._done_action_count = 0
        self._first_pickup_step = -1
        self._first_drop_step = -1
        self._first_door_open_step = -1
        self._first_box_open_step = -1
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
        if type(horizon) is not int or not 0 < horizon <= self._config.max_episode_steps:
            raise RuntimeError("BabyAI returned an unexpected horizon")
        self._mission = facts.mission
        self._horizon = horizon
        self._previous_facts = facts
        signature = _observation_signature(public)
        self._previous_signature = signature
        self._seen_signatures.add(signature)
        self._discovered_object_labels.update(facts.visible_object_labels)
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
        previous_signature = self._previous_signature
        if previous_facts is None or previous_signature is None:
            raise RuntimeError("BabyAI observation history is unavailable")

        position_before = _agent_position(self._environment)
        carried_before = _carried_object(self._environment)
        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("BabyAI returned invalid termination flags")
        number = _number(reward)
        public, facts = _observation(observation)
        if facts.mission != self._mission:
            raise RuntimeError("BabyAI changed its mission during an Episode")
        position_after = _agent_position(self._environment)
        carried_after = _carried_object(self._environment)

        self._steps += 1
        self._action_counts[action] += 1
        signature = _observation_signature(public)
        observation_novel = signature not in self._seen_signatures
        self._seen_signatures.add(signature)
        self._novel_observation_steps += int(observation_novel)
        newly_discovered = tuple(
            label
            for label in facts.visible_object_labels
            if label not in self._discovered_object_labels
        )
        self._discovered_object_labels.update(newly_discovered)

        pickup_attempt = action == 3
        object_picked_up = bool(
            pickup_attempt and carried_before == _NO_OBJECT and carried_after != _NO_OBJECT
        )
        failed_pickup = pickup_attempt and not object_picked_up
        self._pickup_attempts += int(pickup_attempt)
        self._pickup_events += int(object_picked_up)
        self._failed_pickups += int(failed_pickup)
        if object_picked_up and self._first_pickup_step < 0:
            self._first_pickup_step = self._steps

        drop_attempt = action == 4
        object_dropped = bool(
            drop_attempt and carried_before != _NO_OBJECT and carried_after == _NO_OBJECT
        )
        failed_drop = drop_attempt and not object_dropped
        self._drop_attempts += int(drop_attempt)
        self._drop_events += int(object_dropped)
        self._failed_drops += int(failed_drop)
        if object_dropped and self._first_drop_step < 0:
            self._first_drop_step = self._steps

        toggle_attempt = action == 5
        door_opened = bool(
            toggle_attempt
            and previous_facts.front_cell[0] == 4
            and previous_facts.front_cell[2] != 0
            and facts.front_cell[0] == 4
            and facts.front_cell[2] == 0
        )
        door_closed = bool(
            toggle_attempt
            and previous_facts.front_cell[0] == 4
            and previous_facts.front_cell[2] == 0
            and facts.front_cell[0] == 4
            and facts.front_cell[2] != 0
        )
        box_opened = bool(
            toggle_attempt and previous_facts.front_cell[0] == 7 and facts.front_cell[0] != 7
        )
        toggle_effective = door_opened or door_closed or box_opened
        failed_toggle = toggle_attempt and not toggle_effective
        self._toggle_attempts += int(toggle_attempt)
        self._door_open_events += int(door_opened)
        self._door_close_events += int(door_closed)
        self._box_open_events += int(box_opened)
        self._failed_toggles += int(failed_toggle)
        if door_opened and self._first_door_open_step < 0:
            self._first_door_open_step = self._steps
        if box_opened and self._first_box_open_step < 0:
            self._first_box_open_step = self._steps

        blocked_forward = action == 2 and position_after == position_before
        self._blocked_forward_count += int(blocked_forward)
        done_action = action == 6
        self._done_action_count += int(done_action)
        success = bool(terminated and number > 0.0)
        instruction_failure = bool(terminated and not success)
        expected_reward = 1.0 - 0.9 * self._steps / self._horizon if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("BabyAI reward semantics drifted")
        if truncated != (self._steps >= self._horizon):
            raise RuntimeError("BabyAI horizon semantics drifted")

        ineffective_action = bool(
            signature == previous_signature and number == 0.0 and not terminated
        )
        self._ineffective_actions += int(ineffective_action)
        self._cumulative_return += number
        terminal_reason = _terminal_reason(
            success=success,
            instruction_failure=instruction_failure,
            truncated=truncated,
            object_picked_up=object_picked_up,
            object_dropped=object_dropped,
            door_opened=door_opened,
        )
        self._done = terminated or truncated
        self._previous_facts = facts
        self._previous_signature = signature

        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(self._horizon - self._steps, 0),
            "front_object": facts.front_label,
            "front_object_before_action": previous_facts.front_label,
            "carried_object": carried_after,
            "carried_object_before_action": carried_before,
            "visible_object_labels": list(facts.visible_object_labels),
            "visible_object_count": len(facts.visible_object_labels),
            "newly_discovered_object_labels": list(newly_discovered),
            "discovered_object_labels": list[PolicyValue](
                sorted(self._discovered_object_labels)
            ),
            "discovered_object_label_count": len(self._discovered_object_labels),
            "pickup_attempt": pickup_attempt,
            "object_picked_up_this_step": object_picked_up,
            "failed_pickup": failed_pickup,
            "pickup_attempt_count": self._pickup_attempts,
            "pickup_event_count": self._pickup_events,
            "failed_pickup_count": self._failed_pickups,
            "first_pickup_step": self._first_pickup_step,
            "drop_attempt": drop_attempt,
            "object_dropped_this_step": object_dropped,
            "failed_drop": failed_drop,
            "drop_attempt_count": self._drop_attempts,
            "drop_event_count": self._drop_events,
            "failed_drop_count": self._failed_drops,
            "first_drop_step": self._first_drop_step,
            "toggle_attempt": toggle_attempt,
            "toggle_effective": toggle_effective,
            "door_opened_this_step": door_opened,
            "door_closed_this_step": door_closed,
            "box_opened_this_step": box_opened,
            "failed_toggle": failed_toggle,
            "toggle_attempt_count": self._toggle_attempts,
            "door_open_event_count": self._door_open_events,
            "door_close_event_count": self._door_close_events,
            "box_open_event_count": self._box_open_events,
            "failed_toggle_count": self._failed_toggles,
            "first_door_open_step": self._first_door_open_step,
            "first_box_open_step": self._first_box_open_step,
            "blocked_forward": blocked_forward,
            "blocked_forward_count": self._blocked_forward_count,
            "done_action": done_action,
            "done_action_count": self._done_action_count,
            "observation_novel": observation_novel,
            "unique_observation_count": len(self._seen_signatures),
            "observation_novelty_step_fraction": (self._novel_observation_steps / self._steps),
            "ineffective_action": ineffective_action,
            "ineffective_action_fraction": self._ineffective_actions / self._steps,
            "instruction_failure": instruction_failure,
            "task_stage": _task_stage(
                success=success,
                instruction_failure=instruction_failure,
                truncated=truncated,
                carried_object=carried_after,
            ),
            "success_reward_at_this_step": 1.0 - 0.9 * self._steps / self._horizon,
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


def _observation(value: object) -> tuple[dict[str, PolicyValue], _ObservationFacts]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("BabyAI returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
        or numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("BabyAI returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("BabyAI returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("BabyAI returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or not mission:
        raise RuntimeError("BabyAI returned invalid mission")
    public: dict[str, PolicyValue] = {
        "image": TensorValue(
            dtype="uint8",
            shape=_IMAGE_SHAPE,
            data=image.tobytes(order="C"),
        ),
        "direction": direction,
        "mission": mission,
    }
    front_cell = (
        int(image[3, 5, 0]),
        int(image[3, 5, 1]),
        int(image[3, 5, 2]),
    )
    labels = sorted(
        {
            _cell_label(
                (
                    int(image[x, y, 0]),
                    int(image[x, y, 1]),
                    int(image[x, y, 2]),
                )
            )
            for x in range(7)
            for y in range(7)
            if int(image[x, y, 0]) not in {0, 1, 2, 3, 10}
        }
    )
    return public, _ObservationFacts(
        direction=direction,
        mission=mission,
        front_cell=front_cell,
        front_label=_cell_label(front_cell),
        visible_object_labels=tuple(labels),
    )


def _cell_label(cell: tuple[int, int, int]) -> str:
    object_code, color_code, state_code = cell
    kind = _OBJECTS[object_code]
    if kind == "door":
        return f"{_COLORS[color_code]}_{_STATES[state_code]}_door"
    if kind in {"key", "ball", "box"}:
        return f"{_COLORS[color_code]}_{kind}"
    return kind


def _agent_position(environment: gymnasium.Env[object, int]) -> tuple[int, int]:
    value = environment.get_wrapper_attr("agent_pos")
    if (
        not isinstance(value, (tuple, list, numpy.ndarray))
        or len(value) != 2
        or any(type(item) not in {int, numpy.int64} for item in value)
    ):
        raise RuntimeError("BabyAI returned an invalid agent position")
    return int(value[0]), int(value[1])


def _carried_object(environment: gymnasium.Env[object, int]) -> str:
    value = environment.get_wrapper_attr("carrying")
    if value is None:
        return _NO_OBJECT
    kind = getattr(value, "type", None)
    color = getattr(value, "color", None)
    if kind not in {"key", "ball", "box"} or color not in _COLORS:
        raise RuntimeError("BabyAI returned an invalid carried object")
    return f"{color}_{kind}"


def _observation_signature(value: dict[str, PolicyValue]) -> tuple[bytes, int]:
    image = value["image"]
    direction = value["direction"]
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("BabyAI public observation is invalid")
    return image.data, direction


def _terminal_reason(
    *,
    success: bool,
    instruction_failure: bool,
    truncated: bool,
    object_picked_up: bool,
    object_dropped: bool,
    door_opened: bool,
) -> str:
    if success:
        return "success"
    if instruction_failure:
        if object_picked_up:
            return "wrong_pickup_or_order"
        if object_dropped:
            return "wrong_placement_or_order"
        if door_opened:
            return "wrong_door_or_order"
        return "instruction_failure"
    if truncated:
        return "time_limit"
    return "in_progress"


def _task_stage(
    *,
    success: bool,
    instruction_failure: bool,
    truncated: bool,
    carried_object: str,
) -> str:
    if success:
        return "completed"
    if instruction_failure:
        return "failed_instruction"
    if truncated:
        return "time_limit"
    if carried_object != _NO_OBJECT:
        return "carrying_object"
    return "interpreting_and_exploring"


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("BabyAI returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("BabyAI returned non-finite reward")
    return number


__all__ = ["BabyAIEnvironment"]
