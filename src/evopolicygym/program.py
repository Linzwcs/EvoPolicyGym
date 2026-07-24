"""Detached immutable Policy Program snapshots."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ProgramChangedError, ProgramLimitError, ProgramSourceError
from .policy import POLICY_ABI_VERSION

_ENTRYPOINT = "policy.py:make_policy"
_DIGEST_DOMAIN = b"evopolicygym/program/v1\0"
_EXCLUDED_DIRECTORIES = frozenset({".git", "__pycache__"})


@dataclass(frozen=True, slots=True)
class ProgramLimits:
    """Bounds applied while freezing a local Program directory."""

    max_files: int = 1_000
    max_total_bytes: int = 64 * 1024 * 1024
    max_file_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_files", "max_total_bytes", "max_file_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes")


@dataclass(frozen=True, slots=True)
class _ProgramFile:
    path: str
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True, init=False)
class Program:
    """A pathless immutable snapshot of one submitted Policy directory."""

    _files: tuple[_ProgramFile, ...] = field(repr=False)
    _digest: str

    def __init__(self) -> None:
        raise TypeError("Program must be created with Program.from_directory()")

    @classmethod
    def from_directory(
        cls,
        path: str | os.PathLike[str],
        *,
        limits: ProgramLimits | None = None,
    ) -> Program:
        """Freeze a stable directory without retaining its Host path."""

        selected_limits = ProgramLimits() if limits is None else limits
        if type(selected_limits) is not ProgramLimits:
            raise TypeError("limits must be ProgramLimits or None")
        try:
            root = Path(os.fspath(path))
        except TypeError:
            raise TypeError("path must be a path-like string") from None

        try:
            root_stat = root.lstat()
        except OSError:
            raise ProgramSourceError("Program directory cannot be inspected") from None
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise ProgramSourceError("Program source must be a real directory")

        first_paths = _discover_files(root)
        if len(first_paths) > selected_limits.max_files:
            raise ProgramLimitError("Program contains too many files")

        files: list[_ProgramFile] = []
        total_bytes = 0
        for relative_path in first_paths:
            content = _read_stable_file(
                root / relative_path,
                max_bytes=selected_limits.max_file_bytes,
            )
            total_bytes += len(content)
            if total_bytes > selected_limits.max_total_bytes:
                raise ProgramLimitError("Program exceeds its total byte limit")
            files.append(_ProgramFile(path=relative_path, content=content))

        if _discover_files(root) != first_paths:
            raise ProgramChangedError("Program directory changed while being frozen")
        if "policy.py" not in first_paths:
            raise ProgramSourceError("Program must contain policy.py")

        frozen_files = tuple(files)
        value = object.__new__(cls)
        object.__setattr__(value, "_files", frozen_files)
        object.__setattr__(value, "_digest", _program_digest(frozen_files))
        return value

    @property
    def digest(self) -> str:
        """Content identity including the fixed entrypoint and Policy ABI."""

        return self._digest

    @property
    def entrypoint(self) -> str:
        return _ENTRYPOINT

    @property
    def policy_abi(self) -> str:
        return POLICY_ABI_VERSION

    @property
    def files(self) -> tuple[str, ...]:
        """Canonical relative POSIX paths in deterministic order."""

        return tuple(item.path for item in self._files)

    @property
    def file_count(self) -> int:
        return len(self._files)

    @property
    def total_bytes(self) -> int:
        return sum(len(item.content) for item in self._files)

    def read_bytes(self, path: str, /) -> bytes:
        """Read one frozen file by canonical relative path."""

        if type(path) is not str:
            raise TypeError("path must be text")
        for item in self._files:
            if item.path == path:
                return bytes(item.content)
        raise KeyError(path)

    def write_to(self, directory: str | os.PathLike[str]) -> None:
        """Materialize the snapshot into a new directory."""

        try:
            target = Path(os.fspath(directory))
        except TypeError:
            raise TypeError("directory must be a path-like string") from None
        if target.exists() or target.is_symlink():
            raise FileExistsError(str(target))
        parent = target.parent
        if not parent.is_dir():
            raise FileNotFoundError(str(parent))

        temporary = parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o700)
        try:
            for item in self._files:
                destination = temporary.joinpath(*item.path.split("/"))
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(item.content)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def __repr__(self) -> str:
        return (
            "Program("
            f"digest={self.digest!r}, "
            f"file_count={self.file_count}, "
            f"total_bytes={self.total_bytes}"
            ")"
        )


def _discover_files(root: Path) -> tuple[str, ...]:
    discovered: list[str] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            raise ProgramSourceError("Program directory cannot be read") from None
        for entry in entries:
            name = _canonical_component(entry.name)
            if entry.is_symlink():
                raise ProgramSourceError("Program cannot contain symbolic links")
            if entry.is_dir(follow_symlinks=False):
                if name not in _EXCLUDED_DIRECTORIES:
                    visit(Path(entry.path), (*prefix, name))
                continue
            if entry.is_file(follow_symlinks=False):
                if not name.endswith(".pyc"):
                    discovered.append("/".join((*prefix, name)))
                continue
            raise ProgramSourceError("Program cannot contain special files")

    visit(root, ())
    return tuple(sorted(discovered, key=str.encode))


def _canonical_component(name: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or unicodedata.normalize("NFC", name) != name
    ):
        raise ProgramSourceError("Program contains a non-canonical path")
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ProgramSourceError("Program path is not valid UTF-8") from None
    return name


def _read_stable_file(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ProgramChangedError("Program file changed while being frozen") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ProgramSourceError("Program can contain only regular files")
        if before.st_size > max_bytes:
            raise ProgramLimitError("Program file exceeds its byte limit")

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
            raise ProgramLimitError("Program file exceeds its byte limit")

        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError:
            raise ProgramChangedError("Program file changed while being frozen") from None
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        if identity_before != identity_after or identity_after != identity_current:
            raise ProgramChangedError("Program file changed while being frozen")
        return content
    finally:
        os.close(descriptor)


def _program_digest(files: tuple[_ProgramFile, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    for value in (_ENTRYPOINT, POLICY_ABI_VERSION):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(len(files).to_bytes(8, "big"))
    for item in files:
        path = item.path.encode("utf-8")
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(len(item.content).to_bytes(8, "big"))
        digest.update(item.content)
    return f"sha256:{digest.hexdigest()}"


__all__ = ["Program", "ProgramLimits"]
