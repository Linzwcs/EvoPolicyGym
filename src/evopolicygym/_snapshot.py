"""Shared safe file-discovery and stable-read mechanisms for snapshots."""

from __future__ import annotations

import os
import stat
import unicodedata
from collections.abc import Callable
from pathlib import Path

type ErrorFactory = Callable[[str], Exception]


def discover_snapshot_files(
    root: Path,
    *,
    label: str,
    excluded_directories: frozenset[str],
    error: ErrorFactory,
) -> tuple[str, ...]:
    """Discover canonical regular files without following symbolic links."""

    discovered: list[str] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            raise error(f"{label} directory cannot be read") from None
        for entry in entries:
            name = _canonical_component(entry.name, label=label, error=error)
            if entry.is_symlink():
                raise error(f"{label} cannot contain symbolic links")
            if entry.is_dir(follow_symlinks=False):
                if name not in excluded_directories:
                    visit(Path(entry.path), (*prefix, name))
                continue
            if entry.is_file(follow_symlinks=False):
                if not name.endswith(".pyc"):
                    discovered.append("/".join((*prefix, name)))
                continue
            raise error(f"{label} cannot contain special files")

    visit(root, ())
    return tuple(sorted(discovered, key=str.encode))


def read_stable_snapshot_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
    source_error: ErrorFactory,
    changed_error: ErrorFactory,
    limit_error: ErrorFactory,
    retain_executable: bool,
) -> tuple[bytes, bool]:
    """Read one bounded regular file while rejecting identity changes."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise changed_error(
            f"{label} file changed while being frozen"
        ) from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise source_error(f"{label} can contain only regular files")
        if before.st_size > max_bytes:
            raise limit_error(f"{label} file exceeds its byte limit")

        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > max_bytes:
            raise limit_error(f"{label} file exceeds its byte limit")

        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError:
            raise changed_error(
                f"{label} file changed while being frozen"
            ) from None
        identity_before = _identity(before, retain_executable=retain_executable)
        identity_after = _identity(after, retain_executable=retain_executable)
        identity_current = _identity(
            current,
            retain_executable=retain_executable,
        )
        if (
            identity_before != identity_after
            or identity_after != identity_current
        ):
            raise changed_error(
                f"{label} file changed while being frozen"
            )
        return content, bool(before.st_mode & 0o111)
    finally:
        os.close(descriptor)


def _canonical_component(
    name: str,
    *,
    label: str,
    error: ErrorFactory,
) -> str:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or unicodedata.normalize("NFC", name) != name
    ):
        raise error(f"{label} contains a non-canonical path")
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise error(f"{label} path is not valid UTF-8") from None
    return name


def _identity(
    value: os.stat_result,
    *,
    retain_executable: bool,
) -> tuple[int, ...]:
    common = (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
    if retain_executable:
        return (*common[:2], value.st_mode, *common[2:])
    return common


__all__: list[str] = []
