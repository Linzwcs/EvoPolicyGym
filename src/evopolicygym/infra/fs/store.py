"""Filesystem-backed EvoPolicyGym store."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ...core import Eval, OutcomeStatus, PoolKind, Report, Run, Snap, Submit, Trace
from ...metric import measure as measure_code
from ...protocol import feedback as build_feedback
from ...protocol import outcome, record
from ...protocol.agents import stage as stage_agents

_OBS_EXTERNAL_BYTES = 4096


@dataclass(slots=True)
class FileStore:
    """Persist one run under a filesystem directory."""

    root: Path
    dimensions: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, Any] = field(default_factory=dict)
    _opened: datetime | None = field(default=None, init=False, repr=False)
    _evals: list[Eval] = field(default_factory=list, init=False, repr=False)
    _metrics: dict[int, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _locked: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = _absolute(self.root)

    def open(self, run: Run) -> None:
        self._acquire_lock(run)
        try:
            if (self.root / "run.json").exists():
                raise FileExistsError(f"run already exists: {self.root / 'run.json'}")
            self._opened = datetime.now(UTC)
            for path in (
                self.workspace,
                self.checkpoints,
                self.feedback,
                self.logs,
            ):
                path.mkdir(parents=True, exist_ok=True)
            self.versions = dict(self.versions)
            self.versions["agents_md_hash"] = stage_agents(self.agents)
            self.save(run)
            self.emit(
                "run.open",
                key=run.key,
                model=run.model,
                env=run.env,
                exp=run.exp,
                budget=run.budget.limit,
                protocol=run.protocol,
            )
        except Exception:
            self.release_lock()
            raise

    def save(self, run: Run) -> None:
        _write_json(
            self.root / ".evopolicygym" / "state.json",
            {
                "key": run.key,
                "model": run.model,
                "env": run.env,
                "exp": run.exp,
                "protocol": run.protocol,
                "state": run.state.value,
                "budget": {"limit": run.budget.limit, "used": run.budget.used},
                "outcome": run.outcome.value if run.outcome is not None else None,
                "pick": run.pick.scores if run.pick is not None else None,
            },
        )

    def close(self, run: Run) -> None:
        final = self._final_eval()
        error = None
        if run.outcome == OutcomeStatus.error:
            error = {"type": "error", "message": "run failed"}
        result = outcome(run, final, error=error, auxiliary=self._auxiliary(run))
        _write_json(
            self.root / "run.json",
            record(
                run,
                result,
                dimensions=self._dimensions(run),
                timing=self._timing(),
                versions=self.versions,
            ),
        )
        self.save(run)
        self.emit(
            "run.close",
            key=run.key,
            state=run.state.value,
            outcome=run.outcome.value if run.outcome is not None else None,
            budget_used=run.budget.used,
            budget_left=run.budget.left,
            best_submit_index=run.pick.best if run.pick is not None else None,
        )
        self.release_lock()

    def release_lock(self) -> None:
        """Release this process's active run lock, if any."""

        if not self._locked:
            return
        try:
            self.lock.unlink(missing_ok=True)
        finally:
            self._locked = False

    def snap(self, run: Run, submit: Submit) -> Snap:
        source = self.workspace
        source.mkdir(parents=True, exist_ok=True)
        target = self.checkpoints / _submit_dir(submit.index)
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(source, target)
        metrics = measure_code(target)
        self._metrics[submit.index] = metrics
        _write_json(target / "metrics.json", metrics)
        self.emit(
            "submit.snapshot",
            submit_index=submit.index,
            cases=list(submit.cases),
            checkpoint=str(target.relative_to(self.root)),
            source_lines=metrics["source_lines"],
            cyclomatic_total=metrics["cyclomatic_total"],
            tree_hash=metrics["tree_hash"],
        )
        return Snap(
            index=submit.index,
            submit=submit.index,
            ref=str(target.relative_to(self.root)),
            cost=submit.cost,
        )

    def feed(self, run: Run, submit: Submit, report: Report) -> dict[str, Any]:
        data = build_feedback(run, submit, report)
        root = self.feedback / _submit_dir(submit.index)
        _write_episodes(root, run, report)
        if not report.feed.verdict.success:
            _write_text(root / "errors.txt", _errors(report))
        _write_json(root / "summary.json", data)
        self.emit(
            "submit.feedback",
            submit_index=submit.index,
            status=report.feed.verdict.value,
            cost=report.feed.cost,
            n_episodes=len(report.traces),
            mean_return=report.feed.score.mean,
            wall_time_seconds=round(report.wall, 6),
            path=str(root.relative_to(self.root)),
        )
        return data

    def eval(self, run: Run, record: Eval) -> None:
        self._evals.append(record)
        self.emit(
            "eval.record",
            kind=record.kind.value,
            snap=record.snap,
            pool=record.pool,
            mean_return=record.score.mean,
            value=record.score.value,
        )

    def mirror(self, run: Run, snap: Snap) -> None:
        source = self.root / snap.ref
        if not source.exists():
            raise FileNotFoundError(source)
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        shutil.copytree(source, self.workspace, ignore=_checkpoint_metadata)
        self.emit(
            "workspace.mirror",
            submit_index=snap.submit,
            checkpoint=snap.ref,
            workspace=str(self.workspace.relative_to(self.root)),
        )

    def emit(self, event: str, **data: Any) -> None:
        """Append one framework event to `logs/harness.log`."""

        _append_jsonl(
            self.logs / "harness.log",
            {
                "timestamp": _stamp(datetime.now(UTC)),
                "event": event,
                **{key: _clean(value) for key, value in data.items()},
            },
        )

    @property
    def workspace(self) -> Path:
        return self.root / "workspace" / "system"

    @property
    def agents(self) -> Path:
        return self.root / "workspace" / "AGENTS.md"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def feedback(self) -> Path:
        return self.root / "workspace" / "feedback"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def lock(self) -> Path:
        return self.root / ".evopolicygym" / "lock"

    def _acquire_lock(self, run: Run) -> None:
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "pid": os.getpid(),
            "created_at": _stamp(datetime.now(UTC)),
            "key": run.key,
            "model": run.model,
            "env": run.env,
            "exp": run.exp,
        }
        data = json.dumps(body, indent=2, sort_keys=True) + "\n"
        while True:
            try:
                fd = os.open(str(self.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if _stale(self.lock):
                    self.lock.unlink(missing_ok=True)
                    continue
                raise RuntimeError(f"run root is locked: {self.root}") from None
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(data)
            self._locked = True
            return

    def _final_eval(self) -> Eval | None:
        for item in reversed(self._evals):
            if item.kind == PoolKind.final:
                return item
        return None

    def _dimensions(self, run: Run) -> dict[str, Any]:
        values = {"episode_budget": run.budget.limit}
        values.update(self.dimensions)
        return values

    def _timing(self) -> dict[str, Any]:
        end = datetime.now(UTC)
        start = self._opened or end
        return {
            "start_time": _stamp(start),
            "end_time": _stamp(end),
            "wall_time_seconds": max(0.0, (end - start).total_seconds()),
        }

    def _auxiliary(self, run: Run) -> dict[str, Any]:
        if not self._metrics:
            return {}

        submits = sorted(self._metrics)
        best = run.pick.best if run.pick is not None else None
        return {
            "code_metrics_best": self._metrics.get(best) if best is not None else None,
            "code_metrics_by_submit": {
                str(index): self._metrics[index] for index in submits
            },
            "code_metrics_trend": _metric_trend(self._metrics, submits),
        }


def _submit_dir(index: int) -> str:
    return f"submit_{index:03d}"


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path.cwd() / value


def _metric_trend(metrics: dict[int, dict[str, Any]], submits: list[int]) -> dict[str, Any]:
    keys = (
        "source_lines",
        "python_files",
        "functions",
        "classes",
        "cyclomatic_total",
        "cyclomatic_max",
        "tree_hash",
    )
    trend: dict[str, Any] = {"submits": submits}
    for key in keys:
        trend[key] = [metrics[index].get(key) for index in submits]
    return trend


def _checkpoint_metadata(directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {"_meta.json", "metrics.json"}}


def _episode_dir(index: int, run: Run) -> str:
    width = max(3, len(str(run.budget.limit)))
    return f"ep_{index:0{width}d}"


def _write_episodes(root: Path, run: Run, report: Report) -> None:
    if report.first is None or not report.traces:
        return

    for local, trace in enumerate(report.traces):
        episode = root / "episodes" / _episode_dir(report.first + local, run)
        rows = [dict(row) for row in trace.steps]
        _write_observations(episode, rows)
        _write_lines(episode / "trajectory.jsonl", tuple(rows))
        _write_text(episode / "stdout.txt", trace.stdout)
        _write_text(episode / "stderr.txt", trace.stderr)
        if trace.error:
            _write_text(episode / "error.txt", _trace_error(report, trace))


def _write_observations(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows or not any("obs" in row for row in rows):
        return
    if any(row.get("obs") is None for row in rows):
        return

    np = _numpy()
    if np is None:
        return

    observations = [row.get("obs") for row in rows]
    array = _fixed_array(np, observations)
    if array is not None and array.nbytes > _OBS_EXTERNAL_BYTES:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "observations.npy", array)
        for row in rows:
            row["obs"] = None
        return

    arrays = _fixed_dict_arrays(np, observations)
    if arrays and sum(item.nbytes for item in arrays.values()) > _OBS_EXTERNAL_BYTES:
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "observations.npz", **arrays)
        for row in rows:
            row["obs"] = None


def _fixed_array(np: Any, values: list[Any]) -> Any | None:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return None
    if array.dtype == object or array.shape[:1] != (len(values),):
        return None
    return _compact_array(np, array)


def _compact_array(np: Any, array: Any) -> Any:
    if array.dtype.kind in {"u", "i"} and array.size:
        minimum = int(np.nanmin(array))
        maximum = int(np.nanmax(array))
        if 0 <= minimum and maximum <= 255:
            return array.astype("uint8", copy=False)
    return array


def _fixed_dict_arrays(np: Any, values: list[Any]) -> dict[str, Any] | None:
    if not all(isinstance(value, dict) for value in values):
        return None
    keys = tuple(values[0]) if values else ()
    if not keys or any(tuple(value) != keys for value in values):
        return None

    arrays: dict[str, Any] = {}
    for key in keys:
        array = _fixed_array(np, [value[key] for value in values])
        if array is None:
            return None
        arrays[str(key)] = array
    return arrays


def _numpy() -> Any | None:
    try:
        import numpy as np
    except ModuleNotFoundError:
        return None
    return np


def _write_lines(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    data = "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows)
    _write_text(path, data)


def _errors(report: Report) -> str:
    lines = report.feed.errors or (report.feed.verdict.value,)
    return "".join(_error_line(report, _category(line), line) for line in lines)


def _trace_error(report: Report, trace: Trace) -> str:
    message = trace.error or "episode error"
    return _error_line(report, _category(message), message)


def _error_line(report: Report, category: str, message: str) -> str:
    return json.dumps(
        {
            "schema_version": "0.1",
            "timestamp": _stamp(report.completed),
            "category": category,
            "message": message,
            "traceback": None,
        },
        sort_keys=True,
    ) + "\n"


def _category(message: str) -> str:
    return message.split(":", 1)[0]


def _stamp(value: datetime) -> str:
    return (
        value.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, sort_keys=True) + "\n")


def _clean(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _stamp(value)
    if isinstance(value, tuple):
        return [_clean(item) for item in value]
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    return value


def _stale(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    pid = data.get("pid") if isinstance(data, dict) else None
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)
