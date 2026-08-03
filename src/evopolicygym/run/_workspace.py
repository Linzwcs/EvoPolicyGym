"""Run directory layout and active Agent workspace preparation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..errors import AgentRunError
from ..program import Program
from ..skills import AgentSkill


@dataclass(frozen=True, slots=True)
class RunDirectoryPaths:
    root: Path
    workspace: Path
    skills: Path
    program: Path
    analysis: Path
    feedback: Path
    initial: Path
    submissions: Path
    agent: Path
    validation: Path
    assessment: Path
    control: Path
    socket: Path
    events: Path

    @classmethod
    def under(cls, root: Path) -> RunDirectoryPaths:
        workspace = root / "workspace"
        control = root / "control"
        return cls(
            root=root,
            workspace=workspace,
            skills=workspace / "skills",
            program=workspace / "program",
            analysis=workspace / "analysis",
            feedback=workspace / "feedback",
            initial=root / "initial",
            submissions=root / "submissions",
            agent=root / "agent",
            validation=root / "validation",
            assessment=root / "assessment",
            control=control,
            socket=control / "session.sock",
            events=root / "events.jsonl",
        )


class WorkspaceProgramSource:
    """Capture the mutable Program candidate in one active Run workspace."""

    def __init__(self, directory: Path) -> None:
        if not isinstance(directory, Path):
            raise TypeError("directory must be Path")
        self._directory = directory

    def capture(self) -> Program:
        return Program.from_directory(self._directory)


def prepare_run_directory(
    root: Path,
    initial_program: Program,
    *,
    skills: tuple[AgentSkill, ...] = (),
) -> RunDirectoryPaths:
    if not isinstance(root, Path):
        raise TypeError("run directory must be Path")
    if root.exists() or root.is_symlink():
        raise AgentRunError("run_directory must not already exist")
    if not root.parent.is_dir():
        raise AgentRunError("run_directory parent does not exist")
    root.mkdir(mode=0o700)
    paths = RunDirectoryPaths.under(root)
    for directory in (
        paths.workspace,
        paths.analysis,
        paths.feedback,
        paths.initial,
        paths.submissions,
        paths.agent,
        paths.control,
    ):
        directory.mkdir(mode=0o700)
    initial_program.write_to(paths.initial / "program")
    initial_program.write_to(paths.program)
    if skills:
        paths.skills.mkdir(mode=0o700)
        for skill in skills:
            skill.write_to(paths.skills / skill.name)
            _make_tree_read_only(
                paths.skills / skill.name,
                preserve_executable=True,
            )
        os.chmod(paths.skills, 0o500)
    _make_tree_read_only(paths.initial / "program")
    return paths


def remove_control_directory(control: Path) -> None:
    try:
        control.rmdir()
    except OSError:
        pass


def _make_tree_read_only(
    root: Path,
    *,
    preserve_executable: bool = False,
) -> None:
    for directory, _, files in os.walk(root, topdown=False):
        path = Path(directory)
        for name in files:
            file_path = path / name
            mode = (
                0o500
                if preserve_executable
                and file_path.stat().st_mode & 0o111
                else 0o400
            )
            os.chmod(file_path, mode)
        os.chmod(path, 0o500)


__all__: list[str] = []
