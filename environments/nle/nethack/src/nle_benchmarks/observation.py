"""Strict projection of public NLE arrays into the PolicyValue ABI."""

from __future__ import annotations

from collections.abc import Mapping

import numpy
from evopolicygym.policy import PolicyValue, TensorValue

from .constants import BLSTAT_NAMES, CONDITION_BITS, OBSERVATION_KEYS

_DUNGEON_SHAPE = (21, 79)
_INVENTORY_SIZE = 55
_INVENTORY_STRING_SIZE = 80


def project_observation(value: object) -> dict[str, PolicyValue]:
    """Validate one upstream observation and expose only public game state."""

    if type(value) is not dict or set(value) != set(OBSERVATION_KEYS):
        raise RuntimeError("NLE returned an invalid observation mapping")
    document = value

    glyphs = _array(document, "glyphs", "int16", _DUNGEON_SHAPE)
    chars = _array(document, "chars", "uint8", _DUNGEON_SHAPE)
    colors = _array(document, "colors", "uint8", _DUNGEON_SHAPE)
    blstats = _array(document, "blstats", "int64", (len(BLSTAT_NAMES),))
    message = _array(document, "message", "uint8", (256,))
    inv_glyphs = _array(document, "inv_glyphs", "int16", (_INVENTORY_SIZE,))
    inv_strs = _array(
        document,
        "inv_strs",
        "uint8",
        (_INVENTORY_SIZE, _INVENTORY_STRING_SIZE),
    )
    inv_letters = _array(document, "inv_letters", "uint8", (_INVENTORY_SIZE,))
    inv_oclasses = _array(document, "inv_oclasses", "uint8", (_INVENTORY_SIZE,))
    misc = _array(document, "misc", "int32", (3,))

    stats: dict[str, PolicyValue] = {
        name: int(blstats[index]) for index, name in enumerate(BLSTAT_NAMES)
    }
    condition_mask = int(blstats[25])
    stats["conditions"] = [
        name for bit, name in CONDITION_BITS if condition_mask & bit
    ]

    inventory: list[PolicyValue] = []
    for index in range(_INVENTORY_SIZE):
        description = _text(bytes(inv_strs[index]))
        letter_code = int(inv_letters[index])
        if not description and letter_code == 0:
            continue
        if not description or not 1 <= letter_code <= 127:
            raise RuntimeError("NLE returned an invalid inventory entry")
        inventory.append(
            {
                "letter": chr(letter_code),
                "description": description,
                "glyph": int(inv_glyphs[index]),
                "object_class": int(inv_oclasses[index]),
            }
        )

    modes = tuple(int(item) for item in misc)
    if any(item not in {0, 1} for item in modes) or sum(modes) > 1:
        raise RuntimeError("NLE returned invalid public input-mode flags")
    input_mode = (
        "yes_no"
        if modes[0]
        else "get_line"
        if modes[1]
        else "more"
        if modes[2]
        else "normal"
    )

    return {
        "screen": {
            "glyphs": _tensor(glyphs, "int16"),
            "chars": _tensor(chars, "uint8"),
            "colors": _tensor(colors, "uint8"),
        },
        "stats": stats,
        "message": _text(bytes(message)),
        "inventory": inventory,
        "input_mode": input_mode,
    }


def _array(
    document: Mapping[str, object],
    key: str,
    dtype: str,
    shape: tuple[int, ...],
) -> numpy.ndarray:
    value = document[key]
    if (
        type(value) is not numpy.ndarray
        or value.dtype != numpy.dtype(dtype)
        or value.shape != shape
    ):
        raise RuntimeError(f"NLE returned invalid {key}")
    return value


def _tensor(value: numpy.ndarray, dtype: str) -> TensorValue:
    return TensorValue(
        dtype=dtype,
        shape=tuple(value.shape),
        data=numpy.ascontiguousarray(value).tobytes(order="C"),
    )


def _text(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("latin-1")


__all__ = ["project_observation"]
