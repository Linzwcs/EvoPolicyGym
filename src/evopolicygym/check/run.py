"""Run artifact invariant checker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..metric import measure as measure_code


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    path: str
    message: str


@dataclass(frozen=True, slots=True)
class Report:
    root: Path
    issues: tuple[Issue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(slots=True)
class _Check:
    root: Path
    issues: list[Issue] = field(default_factory=list)

    def run(self) -> Report:
        data = self._json(self.root / "run.json")
        if data is None:
            return self._report()

        summaries = self._summaries()
        self._artifacts(data)
        self._agents(data)
        self._budget(data, summaries)
        metrics = self._metrics(data, summaries)
        self._outcome(data, summaries, metrics)
        return self._report()

    def _summaries(self) -> dict[int, dict[str, Any]]:
        feedback = self.root / "workspace" / "feedback"
        legacy = self.root / "feedback"
        if legacy.exists():
            self._issue(
                "feedback_location",
                legacy,
                "agent-visible feedback must live under workspace/feedback",
            )
        if not feedback.exists():
            self._issue("missing_feedback", feedback, "feedback directory is missing")
            return {}

        submits: dict[int, dict[str, Any]] = {}
        for path in sorted(feedback.glob("submit_*")):
            index = _submit_index(path)
            if index is None:
                self._issue("bad_submit_name", path, "submit directory name is invalid")
                continue
            data = self._json(path / "summary.json")
            if data is None:
                continue
            submits[index] = data
            self._summary(index, path, data)

        expected = list(range(len(submits)))
        actual = sorted(submits)
        if actual and actual != expected:
            self._issue(
                "submit_sequence",
                feedback,
                f"submit indexes must be contiguous from 0: got {actual}",
            )
        return submits

    def _summary(self, index: int, path: Path, data: dict[str, Any]) -> None:
        if data.get("submit_index") != index:
            self._issue("summary_index", path, "summary submit_index does not match path")

        status = data.get("status")
        cases = data.get("env_instances")
        episodes = data.get("n_episodes")
        if not isinstance(cases, list):
            self._issue("summary_cases", path, "env_instances must be a list")
            cases = []
        if not isinstance(episodes, int) or episodes < 0:
            self._issue("summary_episodes", path, "n_episodes must be a non-negative int")
            episodes = 0

        if status == "ok":
            returns = data.get("returns")
            lengths = data.get("episode_lengths")
            if not isinstance(returns, list) or len(returns) != episodes:
                self._issue("summary_returns", path, "ok returns must match n_episodes")
            if not isinstance(lengths, list) or len(lengths) != episodes:
                self._issue("summary_lengths", path, "ok episode_lengths must match n_episodes")
            if episodes != len(cases):
                self._issue("summary_cost", path, "ok n_episodes must match env_instances")
            if isinstance(lengths, list):
                self._episodes(path, data, episodes, lengths)
        else:
            if status in {"budget_invalid", "invalid_env_instance"} and episodes != 0:
                self._issue("summary_reject_cost", path, "phase-one rejects must cost 0")

    def _episodes(
        self,
        path: Path,
        data: dict[str, Any],
        episodes: int,
        lengths: list[Any],
    ) -> None:
        first = data.get("first_global_episode")
        if not isinstance(first, int):
            self._issue("first_episode", path, "ok summary requires first_global_episode")
            return

        root = path / "episodes"
        if not root.exists():
            self._issue("missing_episodes", root, "ok submit requires episodes directory")
            return

        dirs = {index: item for item in root.glob("ep_*") if (index := _episode_index(item)) is not None}
        expected = set(range(first, first + episodes))
        if set(dirs) != expected:
            self._issue(
                "episode_sequence",
                root,
                f"episode dirs {sorted(dirs)} do not match expected {sorted(expected)}",
            )

        for local, global_index in enumerate(range(first, first + episodes)):
            ep = dirs.get(global_index)
            if ep is None:
                continue
            self._episode(ep, lengths[local], local, data)

    def _episode(
        self,
        path: Path,
        length: Any,
        local: int,
        data: dict[str, Any],
    ) -> None:
        trajectory = path / "trajectory.jsonl"
        rows: list[dict[str, Any]] | None = None
        if not trajectory.exists():
            self._issue("missing_trajectory", trajectory, "trajectory.jsonl is missing")
        else:
            rows = self._jsonl(trajectory)
            if rows is not None and isinstance(length, int) and len(rows) != length:
                self._issue(
                    "trajectory_length",
                    trajectory,
                    "trajectory rows must match summary episode_lengths",
                )
            if rows is not None and any("obs" in row and row.get("obs") is None for row in rows):
                expected = length if isinstance(length, int) else len(rows)
                self._observations(path, expected)

        for name in ("stdout.txt", "stderr.txt"):
            if not (path / name).exists():
                self._issue("missing_stream", path / name, f"{name} is missing")

        errors = set(_ints(data.get("errors"))) | set(_ints(data.get("timeouts")))
        if local in errors and not (path / "error.txt").exists():
            self._issue("missing_episode_error", path / "error.txt", "error.txt is missing")

    def _observations(self, path: Path, expected: int) -> None:
        npy = path / "observations.npy"
        npz = path / "observations.npz"
        if not npy.exists() and not npz.exists():
            self._issue(
                "missing_observations",
                path,
                "external observations require observations.npy or observations.npz",
            )
            return

        np = _numpy()
        if np is None:
            self._issue("observation_numpy", path, "NumPy is required to validate observations")
            return

        if npy.exists():
            try:
                array = np.load(npy, allow_pickle=False)
            except Exception as exc:  # noqa: BLE001 - checker reports malformed artifacts.
                self._issue("observation_file", npy, f"cannot load observations.npy: {exc}")
                return
            if not hasattr(array, "shape") or not array.shape or array.shape[0] != expected:
                self._issue("observation_rows", npy, "observation rows must match trajectory rows")
            return

        try:
            arrays = np.load(npz, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - checker reports malformed artifacts.
            self._issue("observation_file", npz, f"cannot load observations.npz: {exc}")
            return
        with arrays:
            if not arrays.files:
                self._issue("observation_file", npz, "observations.npz must contain arrays")
                return
            for key in arrays.files:
                array = arrays[key]
                if not hasattr(array, "shape") or not array.shape or array.shape[0] != expected:
                    self._issue(
                        "observation_rows",
                        npz,
                        f"observation array {key!r} rows must match trajectory rows",
                    )
                    return

    def _artifacts(self, data: dict[str, Any]) -> None:
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict):
            self._issue("artifacts", self.root / "run.json", "artifacts must be an object")
            return
        for key, value in artifacts.items():
            if not isinstance(value, str) or value.startswith("/"):
                self._issue(
                    "artifact_path",
                    self.root / "run.json",
                    f"artifact path {key!r} must be relative",
                )
        harness = artifacts.get("logs_harness")
        if isinstance(harness, str) and not (self.root / harness).exists():
            self._issue(
                "missing_harness_log",
                self.root / harness,
                "framework harness log is missing",
            )

    def _agents(self, data: dict[str, Any]) -> None:
        path = self.root / "workspace" / "AGENTS.md"
        if not path.exists():
            self._issue("missing_agents", path, "workspace/AGENTS.md is missing")
            return

        versions = data.get("versions")
        if not isinstance(versions, dict):
            self._issue("versions", self.root / "run.json", "versions must be an object")
            return
        expected = versions.get("agents_md_hash")
        actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            self._issue(
                "agents_hash",
                self.root / "run.json",
                "versions.agents_md_hash does not match workspace/AGENTS.md",
            )

    def _budget(self, data: dict[str, Any], summaries: dict[int, dict[str, Any]]) -> None:
        dimensions = data.get("experiment_dimensions")
        if not isinstance(dimensions, dict):
            self._issue("dimensions", self.root / "run.json", "experiment_dimensions missing")
            return
        budget = dimensions.get("episode_budget")
        if not isinstance(budget, int):
            self._issue("budget", self.root / "run.json", "episode_budget must be an int")
            return

        spent = sum(_int(summary.get("n_episodes")) for summary in summaries.values())
        remaining = _int(summaries[max(summaries)].get("remaining_budget")) if summaries else budget
        if budget != spent + remaining:
            self._issue(
                "budget_conservation",
                self.root,
                f"budget {budget} != spent {spent} + remaining {remaining}",
            )

    def _metrics(
        self,
        data: dict[str, Any],
        summaries: dict[int, dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        root = self.root / "checkpoints"
        expected = {
            index
            for index, summary in summaries.items()
            if not _phase_one_reject(summary.get("status"))
        }
        if not root.exists():
            if expected:
                self._issue("missing_checkpoint", root, "checkpoints directory is missing")
            self._metrics_auxiliary(data, {})
            return {}

        found: dict[int, dict[str, Any]] = {}
        checkpoints: set[int] = set()
        for path in sorted(root.glob("submit_*")):
            if not path.is_dir():
                continue
            index = _submit_index(path)
            if index is None:
                self._issue("bad_checkpoint_name", path, "checkpoint directory name is invalid")
                continue
            checkpoints.add(index)
            metrics = self._checkpoint_metrics(index, path)
            if metrics is not None:
                found[index] = metrics

        for index in sorted(expected - checkpoints):
            self._issue(
                "missing_checkpoint",
                root / f"submit_{index:03d}",
                "accepted submit is missing a checkpoint",
            )
        self._metrics_auxiliary(data, found)
        return found

    def _checkpoint_metrics(self, index: int, checkpoint: Path) -> dict[str, Any] | None:
        path = checkpoint / "metrics.json"
        if not path.exists():
            self._issue("metrics_missing", path, "checkpoint metrics.json is missing")
            return None

        data = self._json(path)
        if data is None:
            return None
        self._metric_schema(path, data)
        expected = measure_code(checkpoint)
        if data != expected:
            self._issue(
                "metrics_mismatch",
                path,
                f"metrics.json does not match recomputed metrics for submit {index}",
            )
        return data

    def _metric_schema(self, path: Path, data: dict[str, Any]) -> None:
        if data.get("schema_version") != "0.1":
            self._issue("metrics_schema", path, "metrics schema_version must be 0.1")

        ints = (
            "files",
            "python_files",
            "bytes",
            "policy_bytes",
            "lines",
            "source_lines",
            "classes",
            "functions",
            "import_count",
            "cyclomatic_total",
            "cyclomatic_max",
            "test_files",
        )
        for key in ints:
            value = data.get(key)
            if not isinstance(value, int) or value < 0:
                self._issue(
                    "metrics_schema",
                    path,
                    f"{key} must be a non-negative int",
                )

        imports = data.get("imports")
        if not isinstance(imports, list) or any(
            not isinstance(item, str) for item in imports
        ):
            self._issue("metrics_schema", path, "imports must be a list of strings")
        elif data.get("import_count") != len(imports):
            self._issue("metrics_schema", path, "import_count must match imports length")

        errors = data.get("parse_errors")
        if not isinstance(errors, list) or any(
            not isinstance(item, str) for item in errors
        ):
            self._issue("metrics_schema", path, "parse_errors must be a list of strings")

        tree_hash = data.get("tree_hash")
        if not isinstance(tree_hash, str) or not tree_hash.startswith("sha256:"):
            self._issue("metrics_schema", path, "tree_hash must be a sha256 digest")

    def _metrics_auxiliary(
        self,
        data: dict[str, Any],
        metrics: dict[int, dict[str, Any]],
    ) -> None:
        outcome = data.get("outcome")
        if not isinstance(outcome, dict):
            return
        auxiliary = outcome.get("auxiliary")
        if not metrics:
            return
        if not isinstance(auxiliary, dict):
            self._issue(
                "metrics_auxiliary",
                self.root / "run.json",
                "outcome auxiliary must be an object",
            )
            return

        submits = sorted(metrics)
        by_submit = auxiliary.get("code_metrics_by_submit")
        if not isinstance(by_submit, dict):
            self._issue(
                "metrics_auxiliary",
                self.root / "run.json",
                "code_metrics_by_submit is missing",
            )
        else:
            indexes = _string_indexes(by_submit)
            if indexes is None or indexes != set(submits):
                self._issue(
                    "metrics_auxiliary",
                    self.root / "run.json",
                    "code_metrics_by_submit keys must match checkpoint submits",
                )
            for index in submits:
                if by_submit.get(str(index)) != metrics[index]:
                    self._issue(
                        "metrics_auxiliary",
                        self.root / "run.json",
                        f"code_metrics_by_submit[{index}] does not match checkpoint metrics",
                    )

        trend = auxiliary.get("code_metrics_trend")
        if not isinstance(trend, dict):
            self._issue(
                "metrics_auxiliary",
                self.root / "run.json",
                "code_metrics_trend is missing",
            )
        else:
            self._metric_trend(trend, metrics, submits)

        best = outcome.get("best_submit_index")
        best_metrics = auxiliary.get("code_metrics_best")
        expected = metrics.get(best) if isinstance(best, int) else None
        if best_metrics != expected:
            self._issue(
                "metrics_best",
                self.root / "run.json",
                "code_metrics_best does not match best_submit_index",
            )

    def _metric_trend(
        self,
        trend: dict[str, Any],
        metrics: dict[int, dict[str, Any]],
        submits: list[int],
    ) -> None:
        if trend.get("submits") != submits:
            self._issue(
                "metrics_auxiliary",
                self.root / "run.json",
                "code_metrics_trend.submits must match checkpoint order",
            )
        keys = (
            "source_lines",
            "python_files",
            "functions",
            "classes",
            "cyclomatic_total",
            "cyclomatic_max",
            "tree_hash",
        )
        for key in keys:
            expected = [metrics[index].get(key) for index in submits]
            if trend.get(key) != expected:
                self._issue(
                    "metrics_auxiliary",
                    self.root / "run.json",
                    f"code_metrics_trend.{key} does not match checkpoint metrics",
                )

    def _outcome(
        self,
        data: dict[str, Any],
        summaries: dict[int, dict[str, Any]],
        metrics: dict[int, dict[str, Any]],
    ) -> None:
        outcome = data.get("outcome")
        if not isinstance(outcome, dict):
            self._issue("outcome", self.root / "run.json", "outcome must be an object")
            return

        status = outcome.get("status")
        ok_submits = {index for index, item in summaries.items() if item.get("status") == "ok"}
        if status == "completed":
            self._completed(outcome, ok_submits, metrics)
        elif status == "no_ok_submit":
            if ok_submits:
                self._issue("no_ok_submit", self.root / "run.json", "no_ok_submit has ok summaries")
            if outcome.get("best_submit_index") is not None:
                self._issue("best_submit", self.root / "run.json", "no_ok_submit best must be null")
        elif status == "error":
            if outcome.get("error") is None:
                self._issue("error_outcome", self.root / "run.json", "error outcome needs details")
        else:
            self._issue("outcome_status", self.root / "run.json", "unknown outcome status")

    def _completed(
        self,
        outcome: dict[str, Any],
        ok_submits: set[int],
        metrics: dict[int, dict[str, Any]],
    ) -> None:
        best = outcome.get("best_submit_index")
        vals = outcome.get("val_scores")
        if not isinstance(best, int):
            self._issue("best_submit", self.root / "run.json", "completed best must be an int")
            return
        if not isinstance(vals, dict):
            self._issue("val_scores", self.root / "run.json", "completed val_scores required")
            return

        val_indexes = {_int(key) for key in vals}
        if best not in val_indexes:
            self._issue("best_submit", self.root / "run.json", "best not in val_scores")
        if val_indexes != ok_submits:
            self._issue(
                "val_scores",
                self.root / "run.json",
                f"val_scores keys {sorted(val_indexes)} do not match ok submits {sorted(ok_submits)}",
            )
        if outcome.get("final_score") is None:
            self._issue("final_score", self.root / "run.json", "completed final_score required")
        if metrics and best not in metrics:
            self._issue("metrics_best", self.root / "run.json", "best checkpoint has no metrics")
        self._mirror(best)

    def _mirror(self, best: int) -> None:
        source = self.root / "checkpoints" / f"submit_{best:03d}"
        target = self.root / "workspace" / "system"
        if not source.exists():
            self._issue("missing_checkpoint", source, "best checkpoint is missing")
            return
        if not target.exists():
            self._issue("missing_workspace", target, "workspace/system is missing")
            return
        if not _same_tree(source, target):
            self._issue("workspace_mirror", target, "workspace/system differs from best checkpoint")

    def _json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            self._issue("missing_json", path, "json file is missing")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._issue("bad_json", path, str(exc))
            return None
        if not isinstance(data, dict):
            self._issue("bad_json", path, "json root must be an object")
            return None
        return data

    def _jsonl(self, path: Path) -> list[dict[str, Any]] | None:
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                self._issue("bad_jsonl", path, f"line {line_no}: {exc}")
                return None
            if not isinstance(row, dict):
                self._issue("bad_jsonl", path, f"line {line_no}: row must be an object")
                return None
            rows.append(row)
        return rows

    def _issue(self, code: str, path: Path, message: str) -> None:
        self.issues.append(Issue(code=code, path=str(path.relative_to(self.root)), message=message))

    def _report(self) -> Report:
        return Report(root=self.root, issues=tuple(self.issues))


def check(root: str | Path) -> Report:
    """Check one run artifact directory."""

    return _Check(Path(root)).run()


def _submit_index(path: Path) -> int | None:
    prefix = "submit_"
    if not path.name.startswith(prefix):
        return None
    value = path.name[len(prefix) :]
    if not value.isdigit():
        return None
    return int(value)


def _episode_index(path: Path) -> int | None:
    prefix = "ep_"
    if not path.name.startswith(prefix):
        return None
    value = path.name[len(prefix) :]
    if not value.isdigit():
        return None
    return int(value)


def _int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _ints(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, int))


def _phase_one_reject(status: Any) -> bool:
    return status in {"budget_invalid", "invalid_env_instance"}


def _numpy() -> Any | None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None
    return np


def _string_indexes(data: dict[str, Any]) -> set[int] | None:
    indexes: set[int] = set()
    for key in data:
        if not isinstance(key, str) or not key.isdigit():
            return None
        indexes.add(int(key))
    return indexes


def _same_tree(left: Path, right: Path) -> bool:
    left_files = _files(left)
    right_files = _files(right)
    if left_files != right_files:
        return False
    for rel in left_files:
        if (left / rel).read_bytes() != (right / rel).read_bytes():
            return False
    return True


def _files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(root)
        if _ignored(rel):
            continue
        files.add(rel)
    return files


def _ignored(path: Path) -> bool:
    return (
        path.name == "_meta.json"
        or path.name == "metrics.json"
        or path.name.endswith(".pyc")
        or "__pycache__" in path.parts
    )
