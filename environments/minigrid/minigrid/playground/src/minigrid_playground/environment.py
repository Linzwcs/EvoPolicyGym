"""One fresh MiniGrid Playground room-coverage Environment per Episode."""

from __future__ import annotations

import operator
from typing import SupportsIndex, cast

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
            gymnasium.make(
                _ENVIRONMENT_ID,
                max_steps=_MAX_EPISODE_STEPS,
            ),
        )
        self._visited_rooms: set[tuple[int, int]] = set()
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        observation, _ = self._environment.reset(seed=self._seed)
        public = _observation(observation)
        self._visited_rooms.add(self._room())
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

        observation, _, upstream_terminated, truncated, _ = (
            self._environment.step(action)
        )
        if (
            type(upstream_terminated) is not bool
            or type(truncated) is not bool
        ):
            raise RuntimeError("MiniGrid Playground returned invalid flags")
        public = _observation(observation)
        room = self._room()
        new_room = room not in self._visited_rooms
        self._visited_rooms.add(room)
        rooms_visited = len(self._visited_rooms)
        success = rooms_visited == _ROOMS
        terminated = upstream_terminated or success
        coverage = rooms_visited / _ROOMS
        self._done = terminated or truncated
        return Step(
            observation=public,
            reward=1.0 if new_room else 0.0,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "rooms_visited": rooms_visited,
                "room_coverage": coverage,
                "new_room": new_room,
                "success": success,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._environment.close()
        self._closed = True

    def _room(self) -> tuple[int, int]:
        value = self._environment.get_wrapper_attr("agent_pos")
        if (
            not isinstance(value, (tuple, list, numpy.ndarray))
            or len(value) != 2
        ):
            raise RuntimeError("MiniGrid Playground returned invalid position")
        try:
            x = operator.index(cast(SupportsIndex, value[0]))
            y = operator.index(cast(SupportsIndex, value[1]))
        except TypeError as error:
            raise RuntimeError(
                "MiniGrid Playground returned invalid position"
            ) from error
        if not 1 <= x <= 17 or not 1 <= y <= 17:
            raise RuntimeError("MiniGrid Playground returned invalid position")
        return min((x - 1) // 6, 2), min((y - 1) // 6, 2)


def _observation(value: object) -> dict[str, PolicyValue]:
    if type(value) is not dict or set(value) != {
        "image",
        "direction",
        "mission",
    }:
        raise RuntimeError("MiniGrid Playground returned invalid observation")
    image = value["image"]
    if (
        type(image) is not numpy.ndarray
        or image.shape != _IMAGE_SHAPE
        or image.dtype != numpy.dtype("uint8")
    ):
        raise RuntimeError("MiniGrid Playground returned invalid image")
    try:
        direction = operator.index(cast(SupportsIndex, value["direction"]))
    except TypeError as error:
        raise RuntimeError(
            "MiniGrid Playground returned invalid direction"
        ) from error
    if not 0 <= direction <= 3:
        raise RuntimeError("MiniGrid Playground returned invalid direction")
    mission = value["mission"]
    if type(mission) is not str or mission != "":
        raise RuntimeError("MiniGrid Playground returned invalid mission")
    return {
        "image": TensorValue(
            dtype="uint8",
            shape=_IMAGE_SHAPE,
            data=image.tobytes(order="C"),
        ),
        "direction": direction,
        "mission": mission,
    }


__all__ = ["PlaygroundEnvironment"]
