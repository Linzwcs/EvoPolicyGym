"""Subprocess-backed runtime adapter."""

from __future__ import annotations

import io
import multiprocessing as mp
import queue
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...core import Exec, Pool, Run, Score, Snap, Submit, Task, Verdict
from .policy import PolicyRuntime, Runner, Value


@dataclass(frozen=True, slots=True)
class Sandbox:
    """Subprocess execution limits."""

    rollout: float | None = None
    memory: int | None = None
    context: str | None = None

    def __post_init__(self) -> None:
        if self.rollout is not None and self.rollout <= 0:
            raise ValueError("sandbox rollout must be positive")
        if self.memory is not None and self.memory <= 0:
            raise ValueError("sandbox memory must be positive")
        if self.context is not None and self.context not in mp.get_all_start_methods():
            raise ValueError(f"unsupported sandbox context: {self.context}")


@dataclass(slots=True)
class SandboxRuntime:
    """Run policy lifecycle calls in short-lived child processes."""

    root: Path
    roller: Runner
    sandbox: Sandbox = field(default_factory=Sandbox)
    denied: frozenset[str] = frozenset()
    value: Value | None = None
    _runs: dict[int, Run] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = _absolute(self.root)

    def scan(self, snap: Snap, task: Task) -> Verdict | None:
        return PolicyRuntime(self.root, self.roller, denied=self.denied).scan(snap, task)

    def load(self, snap: Snap, task: Task) -> Verdict | None:
        result = self._call("load", (snap, task))
        if result.kind == "ok":
            return result.value
        return Verdict.import_error

    def start(
        self,
        run: Run,
        snap: Snap,
        submit: Submit,
        task: Task,
        pool: Pool,
    ) -> Verdict | None:
        self._runs[snap.index] = run
        result = self._call("start", (run, snap, submit, task, pool))
        if result.kind == "ok":
            return result.value
        return Verdict.init_error

    def execute(self, snap: Snap, submit: Submit, task: Task, pool: Pool) -> Exec:
        run = self._runs.get(snap.index)
        if run is None:
            return _exec(Verdict.init_error, "start was not called before execute")

        result = self._call("execute", (run, snap, submit, task, pool))
        if result.kind == "ok":
            return result.value
        if result.kind == "timeout":
            return _exec(Verdict.rollout, "sandbox rollout timeout")
        if result.kind == "crash":
            return _exec(Verdict.oom, result.message)
        if "MemoryError" in result.message:
            return _exec(Verdict.oom, result.message)
        return _exec(Verdict.init_error, result.message)

    def eval(self, snap: Snap, pool: Pool, task: Task) -> Score:
        result = self._call("eval", (snap, pool, task))
        if result.kind == "ok":
            return result.value
        if result.kind == "timeout":
            raise TimeoutError("sandbox rollout timeout")
        raise RuntimeError(result.message)

    def _call(self, op: str, args: tuple[Any, ...]) -> _Result:
        context = _context(self.sandbox.context)
        output = context.Queue(maxsize=1)
        ready = context.Event()
        spec = _Spec(
            root=self.root,
            roller=self.roller,
            denied=self.denied,
            value=self.value,
            memory=self.sandbox.memory,
        )
        process = context.Process(target=_child, args=(output, ready, spec, op, args))
        process.start()
        self._join(process, ready, _timeout(op, self.sandbox))

        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            return _Result(kind="timeout")

        try:
            kind, value = output.get(timeout=0.1)
        except queue.Empty:
            return _Result(kind="crash", message=_crash(process.exitcode))
        if kind == "ok":
            return _Result(kind="ok", value=value)
        return _Result(kind="error", message=value)

    @staticmethod
    def _join(process: mp.Process, ready: Any, timeout: float | None) -> None:
        if timeout is None:
            process.join()
            return
        while process.is_alive() and not ready.is_set():
            process.join(timeout=0.05)
        if process.is_alive():
            process.join(timeout)


@dataclass(frozen=True, slots=True)
class _Spec:
    root: Path
    roller: Runner
    denied: frozenset[str]
    value: Value | None
    memory: int | None


@dataclass(frozen=True, slots=True)
class _Result:
    kind: str
    value: Any = None
    message: str = ""


def _child(output: Any, ready: Any, spec: _Spec, op: str, args: tuple[Any, ...]) -> None:
    try:
        _limit_memory(spec.memory)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            output.put(("ok", _dispatch(spec, ready, op, args)))
    except BaseException:
        output.put(("error", traceback.format_exc()))


def _dispatch(spec: _Spec, ready: Any, op: str, args: tuple[Any, ...]) -> Any:
    runtime = PolicyRuntime(
        spec.root,
        spec.roller,
        denied=spec.denied,
        value=spec.value,
    )
    if op == "load":
        snap, task = args
        return runtime.load(snap, task)
    if op == "start":
        run, snap, submit, task, pool = args
        verdict = runtime.load(snap, task)
        return verdict or runtime.start(run, snap, submit, task, pool)
    if op == "execute":
        run, snap, submit, task, pool = args
        verdict = runtime.load(snap, task)
        if verdict is not None:
            return _exec(verdict, "policy load failed")
        verdict = runtime.start(run, snap, submit, task, pool)
        if verdict is not None:
            return _exec(verdict, "policy start failed")
        ready.set()
        return runtime.execute(snap, submit, task, pool)
    if op == "eval":
        snap, pool, task = args
        return runtime.eval(snap, pool, task, ready=ready.set)
    raise ValueError(f"unknown sandbox op: {op}")


def _exec(verdict: Verdict, message: str) -> Exec:
    return Exec(verdict=verdict, score=Score(mean=None, std=None), errors=(message,))


def _absolute(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return Path.cwd() / value


def _context(name: str | None) -> mp.context.BaseContext:
    if name is not None:
        return mp.get_context(name)
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context()


def _crash(exitcode: int | None) -> str:
    return f"sandbox worker exited without result (exitcode={exitcode})"


def _timeout(op: str, sandbox: Sandbox) -> float | None:
    if op in {"execute", "eval"}:
        return sandbox.rollout
    return None


def _limit_memory(memory: int | None) -> None:
    if memory is None:
        return
    try:
        import resource
    except ImportError:
        return
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
