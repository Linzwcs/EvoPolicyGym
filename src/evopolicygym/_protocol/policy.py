"""Canonical bounded Policy process protocol."""

from __future__ import annotations

import base64
import struct

from ..policy import PolicyValue, TensorValue, copy_policy_value
from ._framing import JsonFrameCodec

POLICY_MAX_FRAME_BYTES = 4 * 1024 * 1024
POLICY_MAX_VALUE_DEPTH = 32
POLICY_MAX_VALUE_ITEMS = 65_536

POLICY_FRAMES = JsonFrameCodec(
    label="Policy",
    max_bytes=POLICY_MAX_FRAME_BYTES,
)


def encode_policy_value(value: PolicyValue) -> object:
    """Encode one admitted PolicyValue into its canonical wire node."""

    admitted = copy_policy_value(value)
    if admitted is None:
        return ["none"]
    if type(admitted) is bool:
        return ["bool", admitted]
    if type(admitted) is int:
        return ["int", str(admitted)]
    if type(admitted) is float:
        encoded = base64.b64encode(struct.pack("!d", admitted)).decode("ascii")
        return ["float", encoded]
    if type(admitted) is str:
        return ["string", admitted]
    if type(admitted) is bytes:
        return ["bytes", base64.b64encode(admitted).decode("ascii")]
    if type(admitted) is TensorValue:
        return [
            "tensor",
            admitted.dtype,
            list(admitted.shape),
            base64.b64encode(admitted.data).decode("ascii"),
        ]
    if type(admitted) is list:
        return ["list", [encode_policy_value(item) for item in admitted]]
    if type(admitted) is tuple:
        return ["tuple", [encode_policy_value(item) for item in admitted]]
    if type(admitted) is dict:
        return [
            "map",
            [
                [key, encode_policy_value(item)]
                for key, item in sorted(
                    admitted.items(),
                    key=lambda pair: pair[0].encode(),
                )
            ],
        ]
    raise TypeError("unsupported PolicyValue")


def decode_policy_value(value: object) -> PolicyValue:
    """Decode and validate one bounded canonical PolicyValue wire node."""

    remaining = [POLICY_MAX_VALUE_ITEMS]
    return _decode_policy_value(value, depth=0, remaining=remaining)


def _decode_policy_value(
    value: object,
    *,
    depth: int,
    remaining: list[int],
) -> PolicyValue:
    if depth > POLICY_MAX_VALUE_DEPTH or remaining[0] <= 0:
        raise ValueError("PolicyValue exceeds its structural limits")
    remaining[0] -= 1
    if type(value) is not list or not value or type(value[0]) is not str:
        raise TypeError("PolicyValue wire node is invalid")
    tag = value[0]
    if tag == "none" and len(value) == 1:
        return None
    if tag == "bool" and len(value) == 2 and type(value[1]) is bool:
        return value[1]
    if tag == "int" and len(value) == 2 and type(value[1]) is str:
        integer = int(value[1])
        if str(integer) != value[1]:
            raise ValueError("PolicyValue integer is not canonical")
        return copy_policy_value(integer)
    if tag == "float" and len(value) == 2 and type(value[1]) is str:
        encoded = base64.b64decode(value[1], validate=True)
        if len(encoded) != 8:
            raise ValueError("PolicyValue float has the wrong size")
        return copy_policy_value(struct.unpack("!d", encoded)[0])
    if tag == "string" and len(value) == 2 and type(value[1]) is str:
        return value[1]
    if tag == "bytes" and len(value) == 2 and type(value[1]) is str:
        return base64.b64decode(value[1], validate=True)
    if tag == "tensor" and len(value) == 4:
        dtype, raw_shape, raw_data = value[1:]
        if (
            type(dtype) is not str
            or type(raw_shape) is not list
            or any(type(size) is not int for size in raw_shape)
            or type(raw_data) is not str
        ):
            raise TypeError("PolicyValue tensor is invalid")
        return TensorValue(
            dtype=dtype,
            shape=tuple(raw_shape),
            data=base64.b64decode(raw_data, validate=True),
        )
    if tag in {"list", "tuple"} and len(value) == 2 and type(value[1]) is list:
        decoded = [
            _decode_policy_value(item, depth=depth + 1, remaining=remaining)
            for item in value[1]
        ]
        return decoded if tag == "list" else tuple(decoded)
    if tag == "map" and len(value) == 2 and type(value[1]) is list:
        result: dict[str, PolicyValue] = {}
        for pair in value[1]:
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or pair[0] in result
            ):
                raise TypeError("PolicyValue map entry is invalid")
            result[pair[0]] = _decode_policy_value(
                pair[1],
                depth=depth + 1,
                remaining=remaining,
            )
        return result
    raise ValueError("PolicyValue wire tag is invalid")


__all__: list[str] = []
