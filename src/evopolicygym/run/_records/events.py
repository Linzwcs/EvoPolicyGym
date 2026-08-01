"""Persisted immutable Run event journal entries."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import TextIO, cast

from ..progress import RunEvent, RunEventValue

_RUN_EVENT_SCHEMA = "evopolicygym/run-event/v1"


def append_event(
    stream: TextIO,
    event: str,
    fields: Mapping[str, object],
) -> RunEvent:
    normalized: dict[str, RunEventValue] = {}
    for name, value in fields.items():
        if type(value) not in {str, int, float, bool, type(None)}:
            raise TypeError("Run event fields must contain JSON scalar values")
        normalized[name] = cast(RunEventValue, value)
    published = RunEvent(
        name=event,
        time_unix_ns=time.time_ns(),
        monotonic_ns=time.monotonic_ns(),
        fields=normalized,
    )
    document = {
        "schema": _RUN_EVENT_SCHEMA,
        "time_unix_ns": published.time_unix_ns,
        "monotonic_ns": published.monotonic_ns,
        "event": published.name,
        **published.fields,
    }
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    stream.write(payload + "\n")
    stream.flush()
    return published


__all__: list[str] = []
