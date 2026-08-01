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

from ._snapshot import discover_snapshot_files, read_stable_snapshot_file
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
        first_paths = discover_snapshot_files(
            root,
            label="Agent Skill",
            excluded_directories=_EXCLUDED_DIRECTORIES,
            error=AgentSkillError,
        )
        if len(first_paths) > selected_limits.max_files:
            raise AgentSkillError("Agent Skill contains too many files")
        if "SKILL.md" not in first_paths:
            raise AgentSkillError("Agent Skill must contain SKILL.md")

        files: list[_AgentSkillFile] = []
        total_bytes = 0
        for relative_path in first_paths:
            content, executable = read_stable_snapshot_file(
                root / relative_path,
                label="Agent Skill",
                max_bytes=(
                    selected_limits.max_instructions_bytes
                    if relative_path == "SKILL.md"
                    else selected_limits.max_file_bytes
                ),
                source_error=AgentSkillError,
                changed_error=AgentSkillError,
                limit_error=AgentSkillError,
                retain_executable=True,
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

        if discover_snapshot_files(
            root,
            label="Agent Skill",
            excluded_directories=_EXCLUDED_DIRECTORIES,
            error=AgentSkillError,
        ) != first_paths:
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
