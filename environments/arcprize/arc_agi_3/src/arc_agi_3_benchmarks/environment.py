"""One complete ARC-AGI-3 game instance per EvoPolicyGym Episode."""

from __future__ import annotations

from typing import Any, cast

import numpy
from arcengine import FrameDataRaw, GameAction, GameState
from evopolicygym.authoring import InvalidAction, Step
from evopolicygym.policy import PolicyValue, TensorValue

from ._upstream import EnvironmentWrapperLike

_FRAME_SHAPE = (64, 64)
# ARCEngine checks its 1,000-frame guard before incrementing the loop counter,
# so a completing action can legally return 1,001 frames.
_MAX_ANIMATION_FRAMES = 1_001
_PALETTE_MIN = 0
_PALETTE_MAX = 15


class ArcAgi3Environment:
    """Strict PolicyValue adapter over one initialized official wrapper."""

    def __init__(
        self,
        wrapper: EnvironmentWrapperLike,
        *,
        game_id: str,
        max_episode_steps: int,
    ) -> None:
        self._wrapper = wrapper
        self._game_id = game_id
        self._max_episode_steps = max_episode_steps
        self._started = False
        self._done = False
        self._closed = False
        self._steps = 0
        self._levels_completed = 0
        self._available_actions: tuple[int, ...] = ()

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        # Arcade.make() initializes the official wrapper by issuing its first
        # RESET. Returning the cached frame avoids an unintended second reset.
        response = self._wrapper.observation_space
        if response is None:
            raise RuntimeError("ARC-AGI-3 did not initialize the game")
        observation = self._observation(response)
        self._levels_completed = response.levels_completed
        self._started = True
        return observation

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        selected, data = _action(action, available=self._available_actions)
        if selected is GameAction.RESET:
            response = self._wrapper.reset()
        else:
            response = self._wrapper.step(selected, data=data)
        if response is None:
            raise RuntimeError("ARC-AGI-3 returned no response")

        previous_levels = self._levels_completed
        observation = self._observation(response)
        self._levels_completed = response.levels_completed
        self._steps += 1
        terminated = response.state is GameState.WIN
        truncated = self._steps >= self._max_episode_steps and not terminated
        self._done = terminated or truncated
        return Step(
            observation=observation,
            reward=float(max(response.levels_completed - previous_levels, 0)),
            terminated=terminated,
            truncated=truncated,
            metrics={
                "state": response.state.value,
                "levels_completed": response.levels_completed,
                "win_levels": response.win_levels,
                "reset": selected is GameAction.RESET,
            },
        )

    def close(self) -> None:
        # EnvironmentWrapper 0.9.9 has no public close method. The shared
        # scorecard is closed by Benchmark.feedback() after every Episode in
        # the evaluation has released its wrapper.
        self._closed = True

    def _observation(self, response: FrameDataRaw) -> PolicyValue:
        _validate_response(response, game_id=self._game_id)
        self._available_actions = tuple(response.available_actions)
        frames = tuple(response.frame)
        return {
            "frames": TensorValue(
                dtype="int8",
                shape=(len(frames), *_FRAME_SHAPE),
                data=b"".join(
                    numpy.ascontiguousarray(frame).tobytes(order="C") for frame in frames
                ),
            ),
            "state": response.state.value,
            "levels_completed": response.levels_completed,
            "win_levels": response.win_levels,
            "available_actions": list(response.available_actions),
        }


def _action(
    value: PolicyValue,
    *,
    available: tuple[int, ...],
) -> tuple[GameAction, dict[str, Any] | None]:
    if type(value) is not dict or "action" not in value:
        raise InvalidAction()
    action_id = value["action"]
    if type(action_id) is not int or not 0 <= action_id <= 7:
        raise InvalidAction()

    if action_id == 6:
        if set(value) != {"action", "x", "y"}:
            raise InvalidAction()
        x = value["x"]
        y = value["y"]
        if type(x) is not int or type(y) is not int or not 0 <= x <= 63 or not 0 <= y <= 63:
            raise InvalidAction()
        data: dict[str, Any] | None = {"x": x, "y": y}
    else:
        if set(value) != {"action"}:
            raise InvalidAction()
        data = None

    if action_id != 0 and action_id not in available:
        raise InvalidAction()
    try:
        selected = GameAction.from_id(action_id)
    except ValueError:
        raise InvalidAction() from None
    return selected, data


def _validate_response(response: FrameDataRaw, *, game_id: str) -> None:
    if type(response) is not FrameDataRaw:
        raise RuntimeError("ARC-AGI-3 returned an invalid response type")
    response_base = response.game_id.split("-", 1)[0]
    expected_base = game_id.split("-", 1)[0]
    if response_base != expected_base:
        raise RuntimeError("ARC-AGI-3 returned the wrong game")
    if type(response.state) is not GameState:
        raise RuntimeError("ARC-AGI-3 returned an invalid game state")
    for name in ("levels_completed", "win_levels"):
        value = getattr(response, name)
        if type(value) is not int or not 0 <= value <= 254:
            raise RuntimeError(f"ARC-AGI-3 returned invalid {name}")
    actions = response.available_actions
    if (
        type(actions) is not list
        or any(type(item) is not int or not 1 <= item <= 7 for item in actions)
        or len(set(actions)) != len(actions)
    ):
        raise RuntimeError("ARC-AGI-3 returned invalid available actions")
    frames = cast(list[object], response.frame)
    if not 1 <= len(frames) <= _MAX_ANIMATION_FRAMES:
        raise RuntimeError("ARC-AGI-3 returned an invalid frame sequence")
    for frame in frames:
        if (
            type(frame) is not numpy.ndarray
            or frame.dtype != numpy.dtype("int8")
            or frame.shape != _FRAME_SHAPE
        ):
            raise RuntimeError("ARC-AGI-3 returned an invalid frame")
        if frame.min() < _PALETTE_MIN or frame.max() > _PALETTE_MAX:
            raise RuntimeError("ARC-AGI-3 returned a frame outside its palette")


__all__ = ["ArcAgi3Environment"]
