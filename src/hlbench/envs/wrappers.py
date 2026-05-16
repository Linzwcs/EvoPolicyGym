"""Observation visibility wrappers."""

from __future__ import annotations

from typing import Any


def jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class PublicObservationWrapper:
    """Convert raw environment output into the policy-visible observation."""

    def __init__(self, mode: str = "jsonable", keys: tuple[str, ...] = ("image", "direction", "mission")) -> None:
        self.mode = mode
        self.keys = keys

    def observation(
        self,
        observation: Any,
        *,
        action_count: int | None = None,
    ) -> Any:
        if self.mode == "minigrid_public" and isinstance(observation, dict):
            public = {key: jsonable(observation.get(key)) for key in self.keys if key in observation}
        elif isinstance(observation, dict):
            public = jsonable(observation)
        else:
            public = jsonable(observation)
        if action_count is not None and isinstance(public, dict):
            public["action_count"] = action_count
        return public

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        return jsonable(info)
