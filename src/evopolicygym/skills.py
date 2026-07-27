"""Pathless immutable Coding Agent Skill snapshots."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AgentSkillError

_DIGEST_DOMAIN = b"evopolicygym/agent-skill/v1\0"
_EXCLUDED_DIRECTORIES = frozenset({".git", "__pycache__"})
_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


@dataclass(frozen=True, slots=True)
class AgentSkillLimits:
    """Bounds applied while freezing one Agent Skill directory."""

    max_files: int = 512
    max_total_bytes: int = 64 * 1024 * 1024
    max_file_bytes: int = 16 * 1024 * 1024
    max_instructions_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_files",
            "max_total_bytes",
            "max_file_bytes",
            "max_instructions_bytes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes")
        if self.max_instructions_bytes > self.max_file_bytes:
            raise ValueError(
                "max_instructions_bytes cannot exceed max_file_bytes"
            )


@dataclass(frozen=True, slots=True)
class _AgentSkillFile:
    path: str
    content: bytes = field(repr=False)
    executable: bool


@dataclass(frozen=True, slots=True, init=False)
class AgentSkill:
    """A detached snapshot of one explicitly selected Coding Agent Skill."""

    _name: str
    _files: tuple[_AgentSkillFile, ...] = field(repr=False)
    _digest: str

    def __init__(self) -> None:
        raise TypeError(
            "AgentSkill must be created with AgentSkill.from_directory()"
        )

    @classmethod
    def from_directory(
        cls,
        path: str | os.PathLike[str],
        *,
        limits: AgentSkillLimits | None = None,
    ) -> AgentSkill:
        """Freeze a standard Skill directory without retaining its Host path."""

        selected_limits = AgentSkillLimits() if limits is None else limits
        if type(selected_limits) is not AgentSkillLimits:
            raise TypeError("limits must be AgentSkillLimits or None")
        try:
            root = Path(os.fspath(path))
        except TypeError:
            raise TypeError("path must be a path-like string") from None

        try:
            root_stat = root.lstat()
        except OSError:
            raise AgentSkillError(
                "Agent Skill directory cannot be inspected"
            ) from None
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(
            root_stat.st_mode
        ):
            raise AgentSkillError(
                "Agent Skill source must be a real directory"
            )

        name = _skill_name(root.name)
        first_paths = _discover_files(root)
        if len(first_paths) > selected_limits.max_files:
            raise AgentSkillError("Agent Skill contains too many files")
        if "SKILL.md" not in first_paths:
            raise AgentSkillError("Agent Skill must contain SKILL.md")

        files: list[_AgentSkillFile] = []
        total_bytes = 0
        for relative_path in first_paths:
            content, executable = _read_stable_file(
                root / relative_path,
                max_bytes=(
                    selected_limits.max_instructions_bytes
                    if relative_path == "SKILL.md"
                    else selected_limits.max_file_bytes
                ),
            )
            total_bytes += len(content)
            if total_bytes > selected_limits.max_total_bytes:
                raise AgentSkillError(
                    "Agent Skill exceeds its total byte limit"
                )
            files.append(
                _AgentSkillFile(
                    path=relative_path,
                    content=content,
                    executable=executable,
                )
            )

        if _discover_files(root) != first_paths:
            raise AgentSkillError(
                "Agent Skill directory changed while being frozen"
            )
        _validate_instructions(name, files)

        frozen_files = tuple(files)
        value = object.__new__(cls)
        object.__setattr__(value, "_name", name)
        object.__setattr__(value, "_files", frozen_files)
        object.__setattr__(
            value,
            "_digest",
            _agent_skill_digest(name, frozen_files),
        )
        return value

    @property
    def name(self) -> str:
        """Return the canonical Skill name and workspace directory name."""

        return self._name

    @property
    def digest(self) -> str:
        """Return the content identity of the complete Skill snapshot."""

        return self._digest

    @property
    def files(self) -> tuple[str, ...]:
        """Return canonical relative POSIX paths in deterministic order."""

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
                destination.parent.mkdir(
                    mode=0o700,
                    parents=True,
                    exist_ok=True,
                )
                with destination.open("xb") as stream:
                    stream.write(item.content)
                os.chmod(destination, 0o700 if item.executable else 0o600)
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def __repr__(self) -> str:
        return (
            "AgentSkill("
            f"name={self.name!r}, "
            f"digest={self.digest!r}, "
            f"file_count={self.file_count}, "
            f"total_bytes={self.total_bytes}"
            ")"
        )


def _skill_name(name: str) -> str:
    if (
        len(name) > 64
        or _NAME_PATTERN.fullmatch(name) is None
        or unicodedata.normalize("NFC", name) != name
    ):
        raise AgentSkillError(
            "Agent Skill directory name must be lowercase hyphen-case"
        )
    return name


def _discover_files(root: Path) -> tuple[str, ...]:
    discovered: list[str] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            raise AgentSkillError(
                "Agent Skill directory cannot be read"
            ) from None
        for entry in entries:
            name = _canonical_component(entry.name)
            if entry.is_symlink():
                raise AgentSkillError(
                    "Agent Skill cannot contain symbolic links"
                )
            if entry.is_dir(follow_symlinks=False):
                if name not in _EXCLUDED_DIRECTORIES:
                    visit(Path(entry.path), (*prefix, name))
                continue
            if entry.is_file(follow_symlinks=False):
                if not name.endswith(".pyc"):
                    discovered.append("/".join((*prefix, name)))
                continue
            raise AgentSkillError(
                "Agent Skill cannot contain special files"
            )

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
        raise AgentSkillError("Agent Skill contains a non-canonical path")
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise AgentSkillError(
            "Agent Skill path is not valid UTF-8"
        ) from None
    return name


def _read_stable_file(path: Path, *, max_bytes: int) -> tuple[bytes, bool]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise AgentSkillError(
            "Agent Skill file changed while being frozen"
        ) from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AgentSkillError(
                "Agent Skill can contain only regular files"
            )
        if before.st_size > max_bytes:
            raise AgentSkillError(
                "Agent Skill file exceeds its byte limit"
            )

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
            raise AgentSkillError(
                "Agent Skill file exceeds its byte limit"
            )

        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError:
            raise AgentSkillError(
                "Agent Skill file changed while being frozen"
            ) from None
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_mode,
            current.st_size,
            current.st_mtime_ns,
        )
        if (
            identity_before != identity_after
            or identity_after != identity_current
        ):
            raise AgentSkillError(
                "Agent Skill file changed while being frozen"
            )
        return content, bool(before.st_mode & 0o111)
    finally:
        os.close(descriptor)


def _validate_instructions(
    name: str,
    files: list[_AgentSkillFile],
) -> None:
    instructions = next(
        item.content for item in files if item.path == "SKILL.md"
    )
    try:
        text = instructions.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise AgentSkillError("SKILL.md must be valid UTF-8") from None
    if not text.strip():
        raise AgentSkillError("SKILL.md must not be empty")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise AgentSkillError("SKILL.md must start with YAML frontmatter")
    try:
        closing = lines.index("---", 1)
    except ValueError:
        raise AgentSkillError(
            "SKILL.md frontmatter is not terminated"
        ) from None
    declared_names = [
        line.partition(":")[2].strip()
        for line in lines[1:closing]
        if line.partition(":")[0].strip() == "name"
    ]
    if len(declared_names) != 1:
        raise AgentSkillError(
            "SKILL.md frontmatter must declare exactly one name"
        )
    declared = declared_names[0]
    if (
        len(declared) >= 2
        and declared[0] == declared[-1]
        and declared[0] in {"'", '"'}
    ):
        declared = declared[1:-1]
    if declared != name:
        raise AgentSkillError(
            "SKILL.md name must match its directory name"
        )


def _agent_skill_digest(
    name: str,
    files: tuple[_AgentSkillFile, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    encoded_name = name.encode("ascii")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(len(files).to_bytes(8, "big"))
    for item in files:
        path = item.path.encode("utf-8")
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(b"\x01" if item.executable else b"\x00")
        digest.update(len(item.content).to_bytes(8, "big"))
        digest.update(item.content)
    return f"sha256:{digest.hexdigest()}"


__all__ = ["AgentSkill", "AgentSkillLimits"]
