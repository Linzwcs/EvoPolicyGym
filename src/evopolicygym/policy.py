"""The complete Policy-author-facing ABI."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

POLICY_ABI_VERSION = "policy/v1"

_DTYPE_FORMATS: dict[str, tuple[int, str | None]] = {
    "bool": (1, None),
    "uint8": (1, None),
    "uint16": (2, None),
    "uint32": (4, None),
    "uint64": (8, None),
    "int8": (1, None),
    "int16": (2, None),
    "int32": (4, None),
    "int64": (8, None),
    "float16": (2, "e"),
    "float32": (4, "f"),
    "float64": (8, "d"),
}


@dataclass(frozen=True, slots=True)
class TensorValue:
    """A canonical dense tensor that can cross the Policy boundary."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if type(self.dtype) is not str or self.dtype not in _DTYPE_FORMATS:
            raise ValueError(f"unsupported tensor dtype: {self.dtype!r}")
        if type(self.shape) is not tuple:
            raise TypeError("tensor shape must be an exact tuple")
        if any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError("tensor dimensions must be non-negative exact integers")
        if type(self.data) is not bytes:
            raise TypeError("tensor data must be exact bytes")

        item_size, float_format = _DTYPE_FORMATS[self.dtype]
        expected_size = math.prod(self.shape) * item_size
        if len(self.data) != expected_size:
            raise ValueError("tensor data size does not match its dtype and shape")
        if self.dtype == "bool" and any(byte not in {0, 1} for byte in self.data):
            raise ValueError("bool tensor data must contain only zero or one")
        if float_format is not None:
            values = struct.iter_unpack(f"<{float_format}", self.data)
            if any(not math.isfinite(value[0]) for value in values):
                raise ValueError("floating tensor data must be finite")


type PolicyScalar = None | bool | int | float | str | bytes
type PolicyValue = (
    PolicyScalar
    | TensorValue
    | list[PolicyValue]
    | tuple[PolicyValue, ...]
    | dict[str, PolicyValue]
)


def copy_policy_value(value: PolicyValue) -> PolicyValue:
    """Validate exact carriers and detach mutable PolicyValue containers."""

    if value is None or type(value) in {bool, str, bytes}:
        return value
    if type(value) is int:
        if not -(2**63) <= value <= 2**64 - 1:
            raise ValueError("PolicyValue integer must fit signed or unsigned 64-bit")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("PolicyValue float must be finite")
        return value
    if type(value) is TensorValue:
        return TensorValue(dtype=value.dtype, shape=value.shape, data=value.data)
    if type(value) is list:
        return [copy_policy_value(item) for item in value]
    if type(value) is tuple:
        return tuple(copy_policy_value(item) for item in value)
    if type(value) is dict:
        copied: dict[str, PolicyValue] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("PolicyValue mapping keys must be exact strings")
            copied[key] = copy_policy_value(item)
        return copied
    raise TypeError(f"unsupported PolicyValue carrier: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Case-independent public information for one fresh Policy instance."""

    observation_space: PolicyValue
    action_space: PolicyValue
    metadata: Mapping[str, PolicyValue]
    policy_seed: int

    def __post_init__(self) -> None:
        if type(self.policy_seed) is not int or not 0 <= self.policy_seed <= 2**64 - 1:
            raise ValueError("policy_seed must be an unsigned 64-bit integer")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        metadata: dict[str, PolicyValue] = {}
        for key, value in self.metadata.items():
            if type(key) is not str:
                raise TypeError("metadata keys must be exact strings")
            metadata[key] = copy_policy_value(value)

        object.__setattr__(
            self,
            "observation_space",
            copy_policy_value(self.observation_space),
        )
        object.__setattr__(self, "action_space", copy_policy_value(self.action_space))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@runtime_checkable
class Policy(Protocol):
    """The only behavior required from a submitted Policy instance."""

    def act(self, observation: PolicyValue) -> PolicyValue:
        """Choose one Action for the current observation."""
        ...


__all__ = [
    "POLICY_ABI_VERSION",
    "Policy",
    "PolicyContext",
    "PolicyScalar",
    "PolicyValue",
    "TensorValue",
    "copy_policy_value",
]
