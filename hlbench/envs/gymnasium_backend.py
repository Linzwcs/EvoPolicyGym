"""Generic Gymnasium environment backend."""

from __future__ import annotations

import math
from typing import Any

from hlbench.core.scenario import ScenarioSpec
from hlbench.core.task import EnvContract
from hlbench.envs.base import StepResult
from hlbench.envs.space_schema import space_to_schema, validate_action
from hlbench.envs.wrappers import PublicObservationWrapper


class GymnasiumEnvInstance:
    def __init__(self, env: Any, wrapper: PublicObservationWrapper) -> None:
        self.env = env
        self.wrapper = wrapper
        self._action_schema = space_to_schema(env.action_space)

    @property
    def action_count(self) -> int | None:
        return int(self.env.action_space.n) if hasattr(self.env.action_space, "n") else None

    @property
    def action_schema(self) -> dict[str, Any]:
        return self._action_schema

    def reset(self, seed: int, config: dict[str, Any] | None = None) -> Any:
        del config
        observation, _info = self.env.reset(seed=seed)
        return self.wrapper.observation(observation, action_count=self.action_count)

    def step(self, action: Any) -> StepResult:
        action = validate_action(self.action_schema, action)
        action = _coerce_action(self.env.action_space, action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        return StepResult(
            observation=self.wrapper.observation(observation, action_count=self.action_count),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self.wrapper.info(info),
        )

    def close(self) -> None:
        self.env.close()


class GymnasiumBackend:
    name = "gymnasium"

    def describe(self, scenario: ScenarioSpec) -> EnvContract:
        env = self._make_raw_env(scenario)
        try:
            reward_range = getattr(env, "reward_range", (None, None))
            low, high = _clean_bound(reward_range[0]), _clean_bound(reward_range[1])
            raw_observation_schema = space_to_schema(env.observation_space)
            action_schema = space_to_schema(env.action_space)
            return EnvContract(
                backend=self.name,
                env_id=scenario.env_id,
                observation_schema=_observation_schema(raw_observation_schema, scenario),
                action_schema=_with_meanings(
                    action_schema,
                    scenario.action_meanings,
                    _action_meaning_key(action_schema),
                ),
                reward_range=(low, high),
                termination={
                    "terminated": "Environment-defined terminal state.",
                    "truncated": f"Episode hit the scenario or environment step limit ({scenario.max_steps}).",
                },
                public_info_schema={"type": "dict", "visibility": "train rollouts only"},
            )
        finally:
            env.close()

    def make(self, scenario: ScenarioSpec) -> GymnasiumEnvInstance:
        return GymnasiumEnvInstance(
            self._make_raw_env(scenario),
            PublicObservationWrapper(mode=scenario.observation_mode),
        )

    def _make_raw_env(self, scenario: ScenarioSpec) -> Any:
        try:
            import gymnasium as gym
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError("Gymnasium backend requires the `gymnasium` package.") from exc
        if scenario.env_id.startswith("MiniGrid-"):
            try:
                import minigrid  # noqa: F401
            except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
                raise RuntimeError("MiniGrid scenarios require the `minigrid` package.") from exc
        try:
            return gym.make(scenario.env_id, **scenario.env_kwargs)
        except ModuleNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - dependency dependent
            raise RuntimeError(f"could not create Gymnasium environment {scenario.env_id!r}: {exc}") from exc


def _with_meanings(schema: dict[str, Any], meanings: tuple[dict[str, Any], ...], key: str) -> dict[str, Any]:
    if not meanings:
        return schema
    merged = dict(schema)
    merged[key] = [dict(item) for item in meanings]
    return merged


def _observation_schema(raw_schema: dict[str, Any], scenario: ScenarioSpec) -> dict[str, Any]:
    if scenario.observation_mode == "image_artifact":
        return {
            "type": "dict",
            "mode": "image_artifact",
            "fields": {
                "type": {"type": "string", "constant": "image"},
                "image_path": {"type": "string", "meaning": "Path to the current image observation artifact."},
                "format": {"type": "string", "constant": "ppm"},
                "shape": {"type": "array", "items": "int", "meaning": "Image shape [height, width, channels]."},
                "dtype": {"type": "string", "meaning": "Original image dtype."},
            },
            "image_schema": _with_meanings(raw_schema, scenario.observation_meanings, "dimensions"),
        }
    if scenario.observation_mode == "minigrid_public":
        return {
            "type": "dict",
            "mode": "minigrid_public",
            "description": "Policy-visible MiniGrid observation dictionary returned to act(observation, context).",
            "fields": {
                "image": raw_schema.get("spaces", {}).get("image", {"type": "unknown"}),
                "direction": raw_schema.get("spaces", {}).get("direction", {"type": "unknown"}),
                "mission": raw_schema.get("spaces", {}).get("mission", {"type": "unknown"}),
                "action_count": {
                    "type": "integer",
                    "meaning": "Number of legal discrete actions exposed by the environment.",
                },
            },
            "image_cell_encoding": _minigrid_cell_encoding(),
            "field_meanings": [dict(item) for item in scenario.observation_meanings],
        }
    return _with_meanings(raw_schema, scenario.observation_meanings, "dimensions")


def _action_meaning_key(schema: dict[str, Any]) -> str:
    return "dimensions" if schema.get("type") == "box" else "actions"


def _minigrid_cell_encoding() -> dict[str, dict[str, int]]:
    try:
        from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
    except Exception:  # pragma: no cover - dependency dependent fallback
        return {
            "object_type": {
                "unseen": 0,
                "empty": 1,
                "wall": 2,
                "floor": 3,
                "door": 4,
                "key": 5,
                "ball": 6,
                "box": 7,
                "goal": 8,
                "lava": 9,
                "agent": 10,
            },
            "color": {
                "red": 0,
                "green": 1,
                "blue": 2,
                "purple": 3,
                "yellow": 4,
                "grey": 5,
            },
            "state": {
                "open": 0,
                "closed": 1,
                "locked": 2,
            },
        }
    return {
        "object_type": {str(key): int(value) for key, value in OBJECT_TO_IDX.items()},
        "color": {str(key): int(value) for key, value in COLOR_TO_IDX.items()},
        "state": {str(key): int(value) for key, value in STATE_TO_IDX.items()},
    }


def _clean_bound(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coerce_action(space: Any, action: Any) -> Any:
    if type(space).__name__ != "Box":
        return action
    import numpy as np

    return np.asarray(action, dtype=space.dtype)
