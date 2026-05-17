"""Workspace layout contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceContract:
    root: Path
    scenario_name: str
    policy_path: Path
    editable_paths: tuple[str, ...] = ("system/", "tools/", "experiments/")
    readonly_paths: tuple[str, ...] = ("AGENTS.md", "task.md", "task_contract.json", "feedback/")

    @property
    def rollout_dir(self) -> Path:
        return self.root / "experiments"

    @property
    def manifest_path(self) -> Path:
        return self.root / "workspace.json"

    def contains_allowed_path(self, path: Path) -> bool:
        try:
            relative = path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return bool(relative.parts) and f"{relative.parts[0]}/" in self.editable_paths

    def to_record(self) -> dict[str, object]:
        record = asdict(self)
        record["root"] = str(self.root)
        record["policy_path"] = str(self.policy_path)
        return record
