"""ProcessExecution selection and Program Evolution graph assembly."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

from ..agents import AgentInvocation, CodingAgent
from ..benchmark import Benchmark, BenchmarkSpec
from ..errors import AgentRunError
from ..program import Program
from ..results import RunResult
from ..skills import AgentSkill
from . import RunConfig, _select_skills
from ._service import ProgramEvolutionRun
from .progress import RunObserver


def run_agent_with_processes(
    initial_program: Program,
    benchmark: Benchmark,
    *,
    agent: CodingAgent,
    run_directory: Path,
    config: RunConfig,
    skills: tuple[AgentSkill, ...] = (),
    observer: RunObserver | None = None,
) -> RunResult:
    from ._task import build_agent_task

    spec = _benchmark_spec(benchmark)
    selected_skills = _select_skills(skills)
    task = build_agent_task(spec, config, selected_skills)
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
        skills=selected_skills,
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
    skills: tuple[AgentSkill, ...] = (),
    observer: RunObserver | None = None,
) -> RunResult:
    """Execute the process-Agent graph used by the public Run and tests."""

    from ..evaluation._service import EvaluationService
    from ..execution.process.agent.runner import (
        ProcessAgentRunner,
        build_agent_environment,
    )
    from ..execution.process.policy.runtime import ProcessPolicyRuntimeFactory
    from ._episode_pool import build_training_episode_pool
    from ._publication import FilesystemSubmissionPublisher
    from ._records.invocation import retain_agent_invocation
    from ._records.recorder import RunDirectoryRecorder
    from ._selection.assessment import ProgramAssessor
    from ._selection.validation import CandidateSelector
    from ._session.gateway import UnixSessionGateway
    from ._session.service import SubmissionSession
    from ._workspace import (
        WorkspaceProgramSource,
        prepare_run_directory,
        remove_control_directory,
    )

    if type(initial_program) is not Program:
        raise TypeError("initial_program must be Program")
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must implement Benchmark")
    if type(invocation) is not AgentInvocation:
        raise TypeError("invocation must be AgentInvocation")
    if type(config) is not RunConfig:
        raise TypeError("config must be RunConfig")
    selected_skills = _select_skills(skills)
    if observer is not None and not isinstance(observer, RunObserver):
        raise TypeError("observer must implement RunObserver or be None")

    selected_spec = _benchmark_spec(benchmark) if spec is None else spec
    if type(selected_spec) is not BenchmarkSpec:
        raise TypeError("spec must be BenchmarkSpec or None")
    episode_pool = build_training_episode_pool(benchmark, config)

    paths = prepare_run_directory(
        run_directory,
        initial_program,
        skills=selected_skills,
    )
    try:
        retain_agent_invocation(paths, invocation)
        with RunDirectoryRecorder(
            paths=paths,
            benchmark_spec=selected_spec,
            initial_program=initial_program,
            config=config,
            agent_identity=invocation.identity,
            skills=selected_skills,
            observer=observer,
        ) as recorder:
            evaluator = EvaluationService(
                policy_runtimes=ProcessPolicyRuntimeFactory(),
                monotonic=monotonic,
            )
            session = SubmissionSession(
                programs=WorkspaceProgramSource(paths.program),
                evaluator=evaluator,
                publisher=FilesystemSubmissionPublisher(
                    submissions_root=paths.submissions,
                    feedback_root=paths.feedback,
                    bulk_retention_bytes=(
                        config.bulk_feedback_retention_bytes
                    ),
                ),
                benchmark=benchmark,
                spec=selected_spec,
                config=config,
                recorder=recorder,
                episode_pool=episode_pool,
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
                candidate_selector=CandidateSelector(
                    evaluator=evaluator,
                    benchmark=benchmark,
                    spec=selected_spec,
                    config=config,
                    recorder=recorder,
                ),
                final_assessor=ProgramAssessor(
                    evaluator=evaluator,
                    benchmark=benchmark,
                    spec=selected_spec,
                    config=config,
                    recorder=recorder,
                ),
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
