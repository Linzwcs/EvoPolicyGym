"""Program Evolution orchestration and process-setting assembly."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import monotonic
from typing import Protocol

from ..agents import AgentInvocation, CodingAgent
from ..benchmark import Benchmark, BenchmarkSpec
from ..errors import AgentRunError
from ..execution.process.agent.runner import AgentExit
from ..program import Program
from ..results import RunResult, RunTerminalReason, SubmissionResult
from . import RunConfig
from .progress import RunObserver


class TerminalSignal(Protocol):
    def wait(self, timeout: float | None = None) -> bool:
        ...


class AgentRunner(Protocol):
    def run(
        self,
        terminal: TerminalSignal,
        *,
        timeout_seconds: float,
    ) -> AgentExit:
        ...


class SessionGateway(Protocol):
    @property
    def terminal(self) -> TerminalSignal:
        ...

    def start(self) -> None:
        ...

    def close(self) -> None:
        ...


class EvolutionSession(Protocol):
    @property
    def submissions(self) -> tuple[SubmissionResult, ...]:
        ...

    @property
    def final_submission_id(self) -> str | None:
        ...

    @property
    def final_program(self) -> Program | None:
        ...

    @property
    def terminal_reason(self) -> RunTerminalReason | None:
        ...

    @property
    def authority_exhausted(self) -> bool:
        ...


class RunRecorder(Protocol):
    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        ...

    def commit(self, result: RunResult, agent_exit: AgentExit) -> None:
        ...


class ProgramEvolutionRun:
    """Coordinate one Agent-driven Program improvement lifecycle."""

    def __init__(
        self,
        *,
        benchmark_id: str,
        initial_program: Program,
        session: EvolutionSession,
        gateway: SessionGateway,
        agent_runner: AgentRunner,
        recorder: RunRecorder,
        agent_timeout_seconds: float,
    ) -> None:
        if type(benchmark_id) is not str or not benchmark_id:
            raise ValueError("benchmark_id must be non-empty text")
        if type(initial_program) is not Program:
            raise TypeError("initial_program must be Program")
        if agent_timeout_seconds <= 0:
            raise ValueError("agent_timeout_seconds must be positive")
        self._benchmark_id = benchmark_id
        self._initial_program = initial_program
        self._session = session
        self._gateway = gateway
        self._agent_runner = agent_runner
        self._recorder = recorder
        self._agent_timeout_seconds = agent_timeout_seconds

    def execute(self) -> RunResult:
        """Execute the Run, commit its result, and return a detached value."""

        agent_exit: AgentExit | None = None
        try:
            self._gateway.start()
            self._recorder.record_event(
                "agent_started",
                {
                    "benchmark_id": self._benchmark_id,
                    "initial_program_digest": self._initial_program.digest,
                },
            )
            agent_exit = self._agent_runner.run(
                self._gateway.terminal,
                timeout_seconds=self._agent_timeout_seconds,
            )
            self._record_agent_exit(agent_exit)
        finally:
            self._gateway.close()

        assert agent_exit is not None
        result = RunResult(
            final_program=self._session.final_program,
            final_submission_id=self._session.final_submission_id,
            submissions=self._session.submissions,
            terminal_reason=_terminal_reason(self._session, agent_exit),
        )
        self._recorder.commit(result, agent_exit)
        return result

    def _record_agent_exit(self, agent_exit: AgentExit) -> None:
        if agent_exit.start_failed:
            fields: dict[str, object] = {}
            if agent_exit.start_error_type is not None:
                fields["error_type"] = agent_exit.start_error_type
            if agent_exit.start_errno is not None:
                fields["errno"] = agent_exit.start_errno
            self._recorder.record_event("agent_start_failed", fields)
            return
        if agent_exit.timed_out:
            self._recorder.record_event("agent_timeout", {})
            return
        if agent_exit.stopped_after_terminal:
            self._recorder.record_event(
                "agent_stopped_after_terminal",
                {"returncode": agent_exit.returncode},
            )
            return
        self._recorder.record_event(
            "agent_exited",
            {"returncode": agent_exit.returncode},
        )


def _terminal_reason(
    session: EvolutionSession,
    agent_exit: AgentExit,
) -> RunTerminalReason:
    if session.terminal_reason is not None:
        return session.terminal_reason
    if (
        agent_exit.timed_out
        or agent_exit.stopped_after_terminal
        or agent_exit.start_failed
        or agent_exit.returncode not in {0, None}
    ):
        return "agent_failed"
    if session.authority_exhausted:
        return "budget_exhausted"
    return "agent_exited"


def run_agent_with_processes(
    initial_program: Program,
    benchmark: Benchmark,
    *,
    agent: CodingAgent,
    run_directory: Path,
    config: RunConfig,
    observer: RunObserver | None = None,
) -> RunResult:
    from ._task import build_agent_task

    spec = _benchmark_spec(benchmark)
    task = build_agent_task(spec, config)
    try:
        invocation = agent.build_invocation(task)
    except AgentRunError:
        raise
    except Exception:
        raise AgentRunError("Coding Agent integration failed") from None
    if type(invocation) is not AgentInvocation:
        raise AgentRunError("Coding Agent returned an invalid invocation")
    if invocation.instructions != task.instructions:
        raise AgentRunError("Coding Agent did not retain the Host task")
    return run_process_agent(
        initial_program,
        benchmark,
        spec=spec,
        invocation=invocation,
        run_directory=run_directory,
        config=config,
        observer=observer,
    )


def run_process_agent(
    initial_program: Program,
    benchmark: Benchmark,
    *,
    invocation: AgentInvocation,
    run_directory: Path,
    config: RunConfig,
    spec: BenchmarkSpec | None = None,
    observer: RunObserver | None = None,
) -> RunResult:
    """Execute the process-Agent graph used by the public Run and tests."""

    from ..evaluation._service import EvaluationService
    from ..execution.process.agent.runner import (
        ProcessAgentRunner,
        build_agent_environment,
    )
    from ..execution.process.policy.runtime import ProcessPolicyRuntimeFactory
    from ._directory import (
        RunDirectoryRecorder,
        WorkspaceProgramSource,
        prepare_run_directory,
        remove_control_directory,
        retain_agent_invocation,
    )
    from ._feedback import FilesystemSubmissionPublisher
    from ._session import SubmissionSession
    from ._socket import UnixSessionGateway

    if type(initial_program) is not Program:
        raise TypeError("initial_program must be Program")
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must implement Benchmark")
    if type(invocation) is not AgentInvocation:
        raise TypeError("invocation must be AgentInvocation")
    if type(config) is not RunConfig:
        raise TypeError("config must be RunConfig")
    if observer is not None and not isinstance(observer, RunObserver):
        raise TypeError("observer must implement RunObserver or be None")

    selected_spec = _benchmark_spec(benchmark) if spec is None else spec
    if type(selected_spec) is not BenchmarkSpec:
        raise TypeError("spec must be BenchmarkSpec or None")

    paths = prepare_run_directory(run_directory, initial_program)
    try:
        retain_agent_invocation(paths, invocation)
        with RunDirectoryRecorder(
            paths=paths,
            benchmark_id=selected_spec.id,
            initial_program=initial_program,
            config=config,
            agent_identity=invocation.identity,
            observer=observer,
        ) as recorder:
            session = SubmissionSession(
                programs=WorkspaceProgramSource(paths.program),
                evaluator=EvaluationService(
                    policy_runtimes=ProcessPolicyRuntimeFactory(),
                    monotonic=monotonic,
                ),
                publisher=FilesystemSubmissionPublisher(
                    submissions_root=paths.submissions,
                    feedback_root=paths.feedback,
                ),
                benchmark=benchmark,
                config=config,
                recorder=recorder,
            )
            gateway = UnixSessionGateway(paths.socket, session)
            runner = ProcessAgentRunner(
                command=invocation.command,
                workspace=paths.workspace,
                environment=build_agent_environment(
                    paths.socket,
                    paths.workspace,
                    inherited_names=invocation.inherited_environment,
                ),
                stdout_path=paths.agent / "stdout.log",
                stderr_path=paths.agent / "stderr.log",
            )
            evolution = ProgramEvolutionRun(
                benchmark_id=selected_spec.id,
                initial_program=initial_program,
                session=session,
                gateway=gateway,
                agent_runner=runner,
                recorder=recorder,
                agent_timeout_seconds=config.agent_timeout_seconds,
            )
            return evolution.execute()
    finally:
        remove_control_directory(paths.control)


def _benchmark_spec(benchmark: Benchmark) -> BenchmarkSpec:
    try:
        spec = benchmark.spec
    except Exception:
        raise AgentRunError("Benchmark specification is unavailable") from None
    if type(spec) is not BenchmarkSpec:
        raise AgentRunError("Benchmark returned an invalid specification")
    return spec


__all__: list[str] = []
