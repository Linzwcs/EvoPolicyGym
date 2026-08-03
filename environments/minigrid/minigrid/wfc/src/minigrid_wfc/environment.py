"""One fresh MiniGrid WFC Environment per Episode."""

from __future__ import annotations

import hashlib
import math
import operator
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from importlib.resources import as_file, files
from typing import SupportsFloat, SupportsIndex, cast

import gymnasium
import numpy
from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue
from minigrid.envs.wfc.config import WFC_PRESETS_ALL
from minigrid.envs.wfc.wfcenv import WFCEnv

from .config import WFCConfig

type Position = tuple[int, int]

_IMAGE_SHAPE = (7, 7, 3)
_ACTIONS = frozenset(range(7))
_MISSION = "traverse the maze to get to the goal"
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
_UNSEEN = 0
_EMPTY = 1
_WALL = 2
_FLOOR = 3
_GOAL = 8
_WALKABLE = frozenset({_EMPTY, _FLOOR, _GOAL})
_GENERATION_ATTEMPTS = 8
_RETRY_DOMAIN = b"evopolicygym-minigrid-wfc/generation-retry/v1\0"
_DIRECTION_VECTORS: tuple[Position, ...] = (
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
)
_ACTION_NAMES = (
    "turn_left",
    "turn_right",
    "move_forward",
    "pick_up",
    "drop",
    "toggle",
    "done",
)
_UNUSED_ACTIONS = frozenset({3, 4, 5, 6})


@dataclass(frozen=True, slots=True)
class _Facts:
    direction: int
    goal_visible: bool
    goal_in_front: bool
    front_object: int
    front_label: str
    visible_wall_count: int
    visible_walkable_count: int
    visible_unseen_count: int
    cells: tuple[tuple[int, int, int], ...]


class WFCEnvironment:
    """Strict seeded adapter around a configured WFC registration."""

    def __init__(self, episode: EpisodeSpec, *, config: WFCConfig) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if type(config) is not WFCConfig:
            raise TypeError("config must be WFCConfig")
        if episode.scenario is not None:
            raise ValueError("WFC configuration belongs in WFCConfig")
        self._seed = episode.environment_seed
        self._config = config
        self._resources = ExitStack()
        upstream = WFC_PRESETS_ALL[config.profile]
        resource = files("minigrid_wfc").joinpath(
            "patterns",
            upstream.pattern_path.name,
        )
        pattern_path = self._resources.enter_context(as_file(resource))
        generation = replace(upstream, pattern_path=pattern_path)
        try:
            self._environment = cast(
                gymnasium.Env[object, int],
                WFCEnv(
                    wfc_config=generation,
                    size=config.size,
                    ensure_connected=True,
                    max_steps=config.max_episode_steps,
                ),
            )
        except BaseException:
            self._resources.close()
            raise
        self._started = False
        self._done = False
        self._closed = False
        self._previous_facts: _Facts | None = None
        self._position: Position = (0, 0)
        self._known_cells: dict[Position, int] = {}
        self._known_goal: Position | None = None
        self._goal_found = False
        self._goal_first_seen_step = -1
        self._steps = 0
        self._last_map_expansion_step = 0
        self._blocked_forward_count = 0
        self._unused_action_count = 0
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
        observation: object | None = None
        for attempt in range(_GENERATION_ATTEMPTS):
            try:
                observation, _ = self._environment.reset(seed=_generation_seed(self._seed, attempt))
                break
            except RuntimeError as error:
                if "Could not generate a valid pattern" not in str(error):
                    raise
        if observation is None:
            raise RuntimeError("MiniGrid WFC exhausted deterministic generation retries")
        public, facts = _observation(observation)
        horizon = self._environment.get_wrapper_attr("max_steps")
        if type(horizon) is not int or horizon != self._config.max_episode_steps:
            raise RuntimeError("MiniGrid WFC returned an unexpected horizon")
        self._integrate(facts)
        self._update_goal_discovery(facts, step=0)
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
            raise RuntimeError("MiniGrid WFC observation history is unavailable")

        observation, reward, terminated, truncated, _ = self._environment.step(action)
        if type(terminated) is not bool or type(truncated) is not bool:
            raise RuntimeError("MiniGrid WFC returned invalid flags")
        number = _number(reward)
        public, facts = _observation(observation)
        self._steps += 1
        signature = _observation_signature(public)
        observation_novel = signature not in self._seen_observation_signatures
        self._seen_observation_signatures.add(signature)
        self._previous_observation_signature = signature
        self._novel_observation_steps += int(observation_novel)
        self._action_counts[action] += 1

        forward_attempt = action == 2
        move_succeeded = bool(forward_attempt and previous_facts.front_object in _WALKABLE)
        blocked_forward = forward_attempt and not move_succeeded
        if move_succeeded:
            delta = _DIRECTION_VECTORS[previous_facts.direction]
            self._position = (
                self._position[0] + delta[0],
                self._position[1] + delta[1],
            )
        self._blocked_forward_count += int(blocked_forward)
        unused_action = action in _UNUSED_ACTIONS
        self._unused_action_count += int(unused_action)

        newly_revealed_cells = self._integrate(facts)
        if newly_revealed_cells:
            self._last_map_expansion_step = self._steps
        self._update_goal_discovery(facts, step=self._steps)
        success = bool(terminated and number > 0.0)
        if success != (forward_attempt and previous_facts.goal_in_front):
            raise RuntimeError("MiniGrid WFC goal semantics drifted")
        if terminated != success:
            raise RuntimeError("MiniGrid WFC termination semantics drifted")
        horizon = self._config.max_episode_steps
        expected_reward = 1.0 - 0.9 * self._steps / horizon if success else 0.0
        if not math.isclose(number, expected_reward, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("MiniGrid WFC reward semantics drifted")
        if truncated != (self._steps == horizon):
            raise RuntimeError("MiniGrid WFC horizon semantics drifted")
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
        known_walkable = sum(code in _WALKABLE for code in self._known_cells.values())
        known_walls = sum(code == _WALL for code in self._known_cells.values())
        known_goal_distance = self._known_goal_distance()
        self._done = terminated or truncated
        self._previous_facts = facts

        metrics: dict[str, PolicyValue] = {
            "step_count": self._steps,
            "remaining_steps": max(horizon - self._steps, 0),
            "goal_visible": facts.goal_visible,
            "goal_found": self._goal_found,
            "goal_first_seen_step": self._goal_first_seen_step,
            "goal_in_front": facts.goal_in_front,
            "goal_in_front_before_action": previous_facts.goal_in_front,
            "front_object": facts.front_label,
            "front_object_before_action": previous_facts.front_label,
            "visible_wall_count": facts.visible_wall_count,
            "visible_walkable_count": facts.visible_walkable_count,
            "visible_unseen_count": facts.visible_unseen_count,
            "newly_revealed_cell_count": newly_revealed_cells,
            "known_cell_count": len(self._known_cells),
            "known_walkable_cell_count": known_walkable,
            "known_wall_cell_count": known_walls,
            "known_frontier_count": self._known_frontier_count(),
            "known_goal_distance": known_goal_distance,
            "known_map_fraction": len(self._known_cells) / (self._config.size**2),
            "last_map_expansion_step": self._last_map_expansion_step,
            "steps_since_map_expansion": self._steps - self._last_map_expansion_step,
            "forward_attempt": forward_attempt,
            "move_succeeded": move_succeeded,
            "blocked_forward": blocked_forward,
            "blocked_forward_count": self._blocked_forward_count,
            "unused_action": unused_action,
            "unused_action_count": self._unused_action_count,
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
        try:
            self._environment.close()
        finally:
            self._resources.close()
            self._closed = True

    def _integrate(self, facts: _Facts) -> int:
        newly_revealed = 0
        forward_x, forward_y = _DIRECTION_VECTORS[facts.direction]
        right_x, right_y = _DIRECTION_VECTORS[(facts.direction + 1) % 4]
        for view_x, view_y, object_code in facts.cells:
            side = view_x - 3
            forward = 6 - view_y
            world = (
                self._position[0] + forward * forward_x + side * right_x,
                self._position[1] + forward * forward_y + side * right_y,
            )
            if world not in self._known_cells:
                newly_revealed += 1
            self._known_cells[world] = object_code
            if object_code == _GOAL:
                self._known_goal = world
        return newly_revealed

    def _update_goal_discovery(self, facts: _Facts, *, step: int) -> None:
        if facts.goal_visible and not self._goal_found:
            self._goal_first_seen_step = step
        self._goal_found = self._goal_found or facts.goal_visible

    def _known_frontier_count(self) -> int:
        return len(
            {
                neighbor
                for position, code in self._known_cells.items()
                if code in _WALKABLE
                for neighbor in _neighbors(position)
                if neighbor not in self._known_cells
            }
        )

    def _known_goal_distance(self) -> int:
        if self._known_goal is None:
            return -1
        queue: deque[tuple[Position, int]] = deque(((self._position, 0),))
        visited = {self._position}
        while queue:
            position, distance = queue.popleft()
            if position == self._known_goal:
                return distance
            for neighbor in _neighbors(position):
                if neighbor in visited or self._known_cells.get(neighbor) not in _WALKABLE:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return -1

    def _task_stage(self, facts: _Facts, *, terminal_reason: str) -> str:
        if terminal_reason != "none":
            return terminal_reason
        if facts.goal_in_front:
            return "enter_goal"
        if self._known_goal is not None:
            return "navigate_to_goal"
        if self._steps - self._last_map_expansion_step >= 100:
            return "exploration_stalled"
        return "explore_maze"


def _observation(value: object) -> tuple[dict[str, PolicyValue], _Facts]:
    if type(value) is not dict or set(value) != {"image", "direction", "mission"}:
        raise RuntimeError("MiniGrid WFC returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid WFC returned invalid image")
    if (
        numpy.any(image[:, :, 0] > 10)
        or numpy.any(image[:, :, 1] > 5)
        or numpy.any(image[:, :, 2] > 2)
    ):
        raise RuntimeError("MiniGrid WFC returned out-of-range image codes")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError("MiniGrid WFC returned invalid direction") from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid WFC returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != _MISSION:
        raise RuntimeError("MiniGrid WFC returned invalid mission")
    objects = image[:, :, 0]
    front_object = int(image[3, 5, 0])
    cells = tuple(
        (view_x, view_y, int(objects[view_x, view_y]))
        for view_x in range(7)
        for view_y in range(7)
        if int(objects[view_x, view_y]) != _UNSEEN
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
            direction=direction,
            goal_visible=bool(numpy.any(objects == _GOAL)),
            goal_in_front=front_object == _GOAL,
            front_object=front_object,
            front_label=_object_label(front_object),
            visible_wall_count=int(numpy.count_nonzero(objects == _WALL)),
            visible_walkable_count=int(numpy.count_nonzero(numpy.isin(objects, tuple(_WALKABLE)))),
            visible_unseen_count=int(numpy.count_nonzero(objects == _UNSEEN)),
            cells=cells,
        ),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, SupportsFloat):
        raise RuntimeError("MiniGrid WFC returned invalid reward")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError("MiniGrid WFC returned non-finite reward")
    return number


def _object_label(code: int) -> str:
    return _OBJECTS[code] if 0 <= code < len(_OBJECTS) else "unknown"


def _observation_signature(observation: dict[str, PolicyValue]) -> tuple[bytes, int]:
    image = observation.get("image")
    direction = observation.get("direction")
    if type(image) is not TensorValue or type(direction) is not int:
        raise RuntimeError("MiniGrid WFC public observation is invalid")
    return image.data, direction


def _neighbors(position: Position) -> tuple[Position, ...]:
    return tuple((position[0] + delta[0], position[1] + delta[1]) for delta in _DIRECTION_VECTORS)


def _generation_seed(seed: int, attempt: int) -> int:
    if attempt == 0:
        return seed
    digest = hashlib.sha256()
    digest.update(_RETRY_DOMAIN)
    digest.update(seed.to_bytes(8, "big"))
    digest.update(attempt.to_bytes(1, "big"))
    return int.from_bytes(digest.digest()[:8], "big")


__all__ = ["WFCEnvironment"]
