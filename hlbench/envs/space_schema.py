"""Serialize and validate common environment spaces."""

from __future__ import annotations

import math
from typing import Any


def space_to_schema(space: Any) -> dict[str, Any]:
    name = type(space).__name__
    if name == "Discrete":
        return {"type": "discrete", "n": int(space.n), "start": int(getattr(space, "start", 0))}
    if name == "Box":
        return {
            "type": "box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": _maybe_scalar(space.low),
            "high": _maybe_scalar(space.high),
        }
    if name == "MultiDiscrete":
        return {"type": "multi_discrete", "nvec": _maybe_scalar(space.nvec), "dtype": str(space.dtype)}
    if name == "MultiBinary":
        return {"type": "multi_binary", "n": _maybe_scalar(space.n), "dtype": str(space.dtype)}
    if name == "Dict":
        return {
            "type": "dict",
            "spaces": {str(key): space_to_schema(value) for key, value in space.spaces.items()},
        }
    if name == "Tuple":
        return {"type": "tuple", "spaces": [space_to_schema(value) for value in space.spaces]}
    return {"type": name.lower()}


def validate_action(schema: dict[str, Any], action: Any) -> Any:
    kind = schema.get("type")
    if kind == "discrete":
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError(f"discrete action must be int, got {type(action).__name__}")
        start = int(schema.get("start", 0))
        n = int(schema["n"])
        if action < start or action >= start + n:
            raise ValueError(f"discrete action {action} outside [{start}, {start + n})")
        return action
    if kind == "box":
        values = _flatten(action)
        low = _broadcast(schema.get("low"), len(values))
        high = _broadcast(schema.get("high"), len(values))
        for index, value in enumerate(values):
            number = float(value)
            lo = low[index]
            hi = high[index]
            if lo is not None and math.isfinite(float(lo)) and number < float(lo):
                raise ValueError(f"box action index {index} below low {lo}")
            if hi is not None and math.isfinite(float(hi)) and number > float(hi):
                raise ValueError(f"box action index {index} above high {hi}")
        expected = _shape_size(schema.get("shape"))
        if expected is not None and expected != len(values):
            raise ValueError(f"box action has {len(values)} values, expected {expected}")
        return action
    if kind == "multi_discrete":
        values = _flatten(action)
        nvec = _flatten(schema["nvec"])
        if len(values) != len(nvec):
            raise ValueError(f"multi_discrete action has {len(values)} values, expected {len(nvec)}")
        for index, value in enumerate(values):
            if int(value) < 0 or int(value) >= int(nvec[index]):
                raise ValueError(f"multi_discrete action index {index} outside [0, {nvec[index]})")
        return action
    if kind == "multi_binary":
        values = _flatten(action)
        expected = _shape_size(schema.get("n")) or int(schema["n"])
        if len(values) != expected:
            raise ValueError(f"multi_binary action has {len(values)} values, expected {expected}")
        if any(int(value) not in (0, 1) for value in values):
            raise ValueError("multi_binary action values must be 0 or 1")
        return action
    if kind == "dict":
        if not isinstance(action, dict):
            raise ValueError("dict action must be a mapping")
        for key, child in schema.get("spaces", {}).items():
            if key not in action:
                raise ValueError(f"missing dict action key {key!r}")
            validate_action(child, action[key])
        return action
    if kind == "tuple":
        if not isinstance(action, (list, tuple)):
            raise ValueError("tuple action must be a list or tuple")
        spaces = schema.get("spaces", [])
        if len(action) != len(spaces):
            raise ValueError(f"tuple action has {len(action)} values, expected {len(spaces)}")
        for value, child in zip(action, spaces, strict=True):
            validate_action(child, value)
        return action
    return action


def _maybe_scalar(value: Any) -> Any:
    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        return _clean(value.item())
    if hasattr(value, "tolist"):
        return _compact_uniform(_clean(value.tolist()))
    return _clean(value)


def _clean(value: Any) -> Any:
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _compact_uniform(value: Any) -> Any:
    flattened = _flatten(value)
    if not flattened:
        return value
    first = flattened[0]
    if all(item == first for item in flattened):
        return first
    return value


def _flatten(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


def _broadcast(value: Any, size: int) -> list[Any | None]:
    if value is None:
        return [None] * size
    values = _flatten(value)
    if len(values) == 1:
        return values * size
    if len(values) != size:
        return [None] * size
    return values


def _shape_size(shape: Any) -> int | None:
    if shape is None:
        return None
    if isinstance(shape, int):
        return shape
    size = 1
    for item in shape:
        size *= int(item)
    return size
