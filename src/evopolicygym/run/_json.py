"""Human-facing JSON projection for public Run values."""

from __future__ import annotations

import base64
from collections.abc import Mapping

from ..policy import TensorValue


def encode_public_json_value(value: object) -> object:
    """Project one public value into its retained JSON representation."""

    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is bytes:
        return {
            "$type": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if type(value) is TensorValue:
        return {
            "$type": "tensor",
            "dtype": value.dtype,
            "shape": list(value.shape),
            "base64": base64.b64encode(value.data).decode("ascii"),
        }
    if type(value) is list:
        return [encode_public_json_value(item) for item in value]
    if type(value) is tuple:
        return {
            "$type": "tuple",
            "items": [encode_public_json_value(item) for item in value],
        }
    if isinstance(value, Mapping):
        return {
            key: encode_public_json_value(item)
            for key, item in value.items()
        }
    raise TypeError(f"unsupported public value: {type(value).__name__}")


__all__: list[str] = []
