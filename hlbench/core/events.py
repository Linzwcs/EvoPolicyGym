"""JSONL event logging."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    name: str
    time: float = field(default_factory=time.time)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = {"time": self.time, "event": self.name}
        record.update(self.payload)
        return record


class EventLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def log(self, name: str, **payload: Any) -> Event:
        event = Event(name=name, payload=payload)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(event.to_record(), sort_keys=True) + "\n")
        return event

    @staticmethod
    def asdict(event: Event) -> dict[str, Any]:
        return asdict(event)
