"""JSON-safe Gymnasium space and value codecs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

try:  # Gymnasium depends on numpy, but keep this module importable without extras.
    import numpy as np
except Exception:  # pragma: no cover - exercised only in minimal dependency envs.
    np = None

Json = Any


def schema(space: Any) -> dict[str, Json]:
    """Return an agent-visible schema for a Gymnasium space."""

    kind = _kind(space)
    if kind in {"Box", "AnyBox"}:
        if _image(space):
            return {
                "type": "Image",
                "shape": _shape(space.shape),
                "dtype": str(space.dtype),
                "layout": "height_width_channels",
                "value_range": [0, 255],
                "storage": "external",
            }
        return {
            "type": "Box",
            "shape": _shape(space.shape),
            "dtype": str(space.dtype),
            "low": _bounds(space.low),
            "high": _bounds(space.high),
        }
    if kind == "Discrete":
        body: dict[str, Json] = {"type": "Discrete", "n": int(space.n)}
        start = int(getattr(space, "start", 0))
        if start:
            body["start"] = start
        return body
    if kind == "MultiDiscrete":
        body = {"type": "MultiDiscrete", "nvec": encode(space.nvec), "dtype": str(space.dtype)}
        start = getattr(space, "start", None)
        if start is not None:
            encoded = encode(start)
            if _nonzero(encoded):
                body["start"] = encoded
        return body
    if kind == "MultiBinary":
        return {"type": "MultiBinary", "n": encode(space.n)}
    if kind == "Tuple":
        return {"type": "Tuple", "spaces": [schema(item) for item in space.spaces]}
    if kind == "Dict":
        return {
            "type": "Dict",
            "spaces": {str(key): schema(item) for key, item in space.spaces.items()},
        }
    if kind == "Text":
        return {
            "type": "Text",
            "min_length": int(getattr(space, "min_length", 0)),
            "max_length": int(getattr(space, "max_length", 0)),
        }
    if kind == "Unicode":
        return {
            "type": "Text",
            "min_length": 0,
            "description": "Unicode action string",
        }
    if kind == "Float":
        return {"type": "Box", "shape": [], "dtype": "float", "low": "-inf", "high": "inf"}
    if kind == "Integer":
        return {"type": "Discrete", "n": "unbounded"}
    if kind == "AnyDict":
        return {"type": "Dict", "additional_properties": True}
    if kind == "Anything":
        return {"type": "Any"}
    return {"type": kind, "repr": repr(space)}


def encode(value: Any) -> Json:
    """Convert observations, actions, and info payloads to JSON-safe values."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return _float(value)
    if hasattr(value, "tolist"):
        return encode(value.tolist())
    if hasattr(value, "item"):
        try:
            return encode(value.item())
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [encode(item) for item in value]
    return repr(value)


def sample(space: Any) -> Json:
    """Sample a JSON-safe action from a Gymnasium action space."""

    kind = _kind(space)
    if kind == "Unicode":
        return "noop()"
    if kind == "Float":
        return 0.0
    if kind == "Integer":
        return 0
    if kind == "AnyDict":
        return {}
    if kind == "Anything":
        return None
    return encode(space.sample())


def action(space: Any, value: Any) -> tuple[Any, bool]:
    """Coerce an agent action into the Gymnasium action space.

    The returned boolean marks whether coercion changed or repaired an invalid
    action. Invalid actions are made deterministic where practical instead of
    raising out of the rollout loop.
    """

    kind = _kind(space)
    if kind == "Discrete":
        return _discrete(space, value)
    if kind == "Box":
        return _box(space, value)
    if kind == "MultiDiscrete":
        return _multi_discrete(space, value)
    if kind == "MultiBinary":
        return _multi_binary(space, value)
    if kind == "Tuple":
        raw = _sequence(value)
        values: list[Any] = []
        invalid = len(raw) != len(space.spaces)
        for index, item in enumerate(space.spaces):
            child, changed = action(item, raw[index] if index < len(raw) else None)
            values.append(child)
            invalid = invalid or changed
        return tuple(values), invalid
    if kind == "Dict":
        raw = value if isinstance(value, Mapping) else {}
        values = {}
        invalid = not isinstance(value, Mapping)
        for key, item in space.spaces.items():
            child, changed = action(item, raw.get(key))
            values[key] = child
            invalid = invalid or changed
        return values, invalid
    if kind == "Unicode":
        if isinstance(value, str):
            return value, False
        return "", True
    if kind == "Float":
        try:
            return float(value), False
        except (TypeError, ValueError):
            return 0.0, True
    if kind == "Integer":
        try:
            return int(value), False
        except (TypeError, ValueError):
            return 0, True
    if kind == "AnyDict":
        if isinstance(value, Mapping):
            return dict(value), False
        return {}, True
    return value, False


def _discrete(space: Any, value: Any) -> tuple[int, bool]:
    start = int(getattr(space, "start", 0))
    stop = start + int(space.n) - 1
    try:
        item = int(value)
    except (TypeError, ValueError):
        return start, True
    if item < start:
        return start, True
    if item > stop:
        return stop, True
    return item, False


def _box(space: Any, value: Any) -> tuple[Any, bool]:
    if np is None:
        return value, False
    invalid = False
    try:
        item = np.asarray(value, dtype=space.dtype)
    except (TypeError, ValueError):
        item = np.zeros(space.shape, dtype=space.dtype)
        invalid = True
    if tuple(item.shape) != tuple(space.shape):
        try:
            item = np.broadcast_to(item, space.shape).astype(space.dtype)
        except ValueError:
            item = np.zeros(space.shape, dtype=space.dtype)
            invalid = True
    clipped = np.clip(item, space.low, space.high).astype(space.dtype)
    invalid = invalid or not bool(np.array_equal(item, clipped))
    return clipped, invalid


def _multi_discrete(space: Any, value: Any) -> tuple[Any, bool]:
    if np is None:
        return value, False
    start = np.asarray(getattr(space, "start", np.zeros_like(space.nvec)), dtype=space.dtype)
    low = start
    high = start + np.asarray(space.nvec, dtype=space.dtype) - 1
    invalid = False
    try:
        item = np.asarray(value, dtype=space.dtype)
    except (TypeError, ValueError):
        item = low.copy()
        invalid = True
    if tuple(item.shape) != tuple(space.nvec.shape):
        try:
            item = np.broadcast_to(item, space.nvec.shape).astype(space.dtype)
        except ValueError:
            item = low.copy()
            invalid = True
    clipped = np.clip(item, low, high).astype(space.dtype)
    invalid = invalid or not bool(np.array_equal(item, clipped))
    return clipped, invalid


def _multi_binary(space: Any, value: Any) -> tuple[Any, bool]:
    if np is None:
        return value, False
    shape = space.shape if hasattr(space, "shape") else (space.n,)
    invalid = False
    try:
        item = np.asarray(value, dtype=int)
    except (TypeError, ValueError):
        item = np.zeros(shape, dtype=int)
        invalid = True
    if tuple(item.shape) != tuple(shape):
        try:
            item = np.broadcast_to(item, shape).astype(int)
        except ValueError:
            item = np.zeros(shape, dtype=int)
            invalid = True
    clipped = (item > 0).astype(space.dtype)
    invalid = invalid or not bool(np.array_equal(item, clipped))
    return clipped, invalid


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return list(value)
    return []


def _bounds(value: Any) -> Json:
    if np is None:
        return encode(value)
    item = np.asarray(value)
    if item.size > 32:
        return {
            "shape": _shape(item.shape),
            "min": _float(float(np.nanmin(item))),
            "max": _float(float(np.nanmax(item))),
        }
    return encode(item)


def _shape(value: Any) -> list[int]:
    return [int(item) for item in tuple(value)]


def _kind(space: Any) -> str:
    return type(space).__name__


def _image(space: Any) -> bool:
    if np is None:
        return False
    shape = tuple(getattr(space, "shape", ()))
    if len(shape) not in {2, 3}:
        return False
    dtype = getattr(space, "dtype", None)
    if np.dtype(dtype) != np.dtype("uint8"):
        return False
    low = np.asarray(getattr(space, "low", None))
    high = np.asarray(getattr(space, "high", None))
    return bool(low.size and high.size and np.nanmin(low) >= 0 and np.nanmax(high) <= 255)


def _float(value: float) -> float | str:
    if math.isfinite(value):
        return float(value)
    return str(value)


def _nonzero(value: Json) -> bool:
    if isinstance(value, list):
        return any(_nonzero(item) for item in value)
    return bool(value)
