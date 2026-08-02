"""Durable atomic file-writing primitives for Run records."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from ...errors import AgentRunError


def write_json_atomic(
    path: Path,
    document: dict[str, object],
    *,
    error_message: str = "Run record could not be committed",
) -> None:
    payload = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8", errors="strict")
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    except OSError as error:
        raise AgentRunError(error_message) from error
    finally:
        temporary.unlink(missing_ok=True)


def write_read_only_text_file(
    path: Path,
    content: str,
    *,
    error_message: str,
) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o400)
    except OSError as error:
        raise AgentRunError(error_message) from error


__all__: list[str] = []
