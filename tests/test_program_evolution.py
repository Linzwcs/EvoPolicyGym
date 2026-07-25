from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from evopolicygym.errors import EvaluationError
from evopolicygym.execution.process.agent.runner import AgentExit
from evopolicygym.program import Program
from evopolicygym.results import (
    AssessmentResult,
    EpisodeSummary,
    Feedback,
    RunResult,
    RunTerminalReason,
    SubmissionResult,
)
from evopolicygym.run._service import (
    ProgramEvolutionRun,
    TerminalSignal,
)
from evopolicygym.run._validation import CandidateSelection


class FakeTerminal:
    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        return False


class FakeGateway:
    def __init__(self) -> None:
        self.terminal = FakeTerminal()
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


class FakeAgentRunner:
    def __init__(self, outcome: AgentExit) -> None:
        self.outcome = outcome
        self.calls = 0

    def run(
        self,
        terminal: TerminalSignal,
        *,
        timeout_seconds: float,
    ) -> AgentExit:
        del terminal, timeout_seconds
        self.calls += 1
        return self.outcome


class FakeSession:
    def __init__(
        self,
        *,
        terminal_reason: RunTerminalReason | None = None,
        authority_exhausted: bool = False,
        submissions: tuple[SubmissionResult, ...] = (),
        candidate_submission_ids: tuple[str, ...] = (),
    ) -> None:
        self.submissions = submissions
        self.candidate_submission_ids = candidate_submission_ids
        self.terminal_reason = terminal_reason
        self.authority_exhausted = authority_exhausted


class FakeCandidateSelector:
    def __init__(
        self,
        selection: CandidateSelection | None = None,
        *,
        gateway: FakeGateway | None = None,
        fail: bool = False,
    ) -> None:
        self.selection = selection
        self.gateway = gateway
        self.fail = fail
        self.calls = 0

    def select(
        self,
        submissions: tuple[SubmissionResult, ...],
        candidate_submission_ids: tuple[str, ...],
    ) -> CandidateSelection:
        del submissions, candidate_submission_ids
        self.calls += 1
        if self.gateway is not None:
            assert self.gateway.closed
        if self.fail:
            raise EvaluationError("trusted fixture failure")
        if self.selection is None:
            raise AssertionError("candidate selection was not expected")
        return self.selection


class FakeFinalAssessor:
    def __init__(
        self,
        result: AssessmentResult | None = None,
        *,
        gateway: FakeGateway | None = None,
        fail: bool = False,
    ) -> None:
        self.result = result
        self.gateway = gateway
        self.fail = fail
        self.calls: list[str] = []

    def assess(
        self,
        submission: SubmissionResult,
    ) -> AssessmentResult | None:
        self.calls.append(submission.submission_id)
        if self.gateway is not None:
            assert self.gateway.closed
        if self.fail:
            raise EvaluationError("trusted Assessment fixture failure")
        return self.result


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []
        self.committed: tuple[RunResult, AgentExit] | None = None

    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        self.events.append((event, dict(fields)))

    def commit(self, result: RunResult, agent_exit: AgentExit) -> None:
        self.committed = (result, agent_exit)


def make_program(root: Path) -> Program:
    source = root / "program"
    source.mkdir()
    (source / "policy.py").write_text(
        "def make_policy(context):\n    return object()\n",
        encoding="utf-8",
    )
    return Program.from_directory(source)


class ProgramEvolutionRunTests(unittest.TestCase):
    def test_clean_agent_exit_produces_agent_exited_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = FakeSession()
            gateway = FakeGateway()
            runner = FakeAgentRunner(AgentExit(returncode=0))
            recorder = FakeRecorder()

            result = ProgramEvolutionRun(
                benchmark_id="example/benchmark-v1",
                initial_program=program,
                session=session,
                gateway=gateway,
                agent_runner=runner,
                candidate_selector=FakeCandidateSelector(),
                final_assessor=FakeFinalAssessor(),
                recorder=recorder,
                agent_timeout_seconds=10,
            ).execute()

        self.assertEqual(result.terminal_reason, "agent_exited")
        self.assertTrue(gateway.started)
        self.assertTrue(gateway.closed)
        self.assertEqual(runner.calls, 1)
        self.assertEqual(
            [event for event, _ in recorder.events],
            ["agent_started", "agent_exited"],
        )
        self.assertIsNotNone(recorder.committed)

    def test_session_terminal_reason_is_authoritative_after_agent_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = FakeSession(terminal_reason="evaluation_failed")
            gateway = FakeGateway()
            runner = FakeAgentRunner(
                AgentExit(
                    returncode=-15,
                    stopped_after_terminal=True,
                )
            )
            recorder = FakeRecorder()

            result = ProgramEvolutionRun(
                benchmark_id="example/benchmark-v1",
                initial_program=program,
                session=session,
                gateway=gateway,
                agent_runner=runner,
                candidate_selector=FakeCandidateSelector(),
                final_assessor=FakeFinalAssessor(),
                recorder=recorder,
                agent_timeout_seconds=10,
            ).execute()

        self.assertEqual(result.terminal_reason, "evaluation_failed")
        self.assertTrue(gateway.closed)
        self.assertEqual(
            recorder.events[-1],
            (
                "agent_stopped_after_terminal",
                {"returncode": -15},
            ),
        )

    def test_agent_start_failure_is_typed_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            recorder = FakeRecorder()

            result = ProgramEvolutionRun(
                benchmark_id="example/benchmark-v1",
                initial_program=program,
                session=FakeSession(),
                gateway=FakeGateway(),
                agent_runner=FakeAgentRunner(
                    AgentExit(
                        returncode=None,
                        start_failed=True,
                        start_error_type="FileNotFoundError",
                        start_errno=2,
                    )
                ),
                candidate_selector=FakeCandidateSelector(),
                final_assessor=FakeFinalAssessor(),
                recorder=recorder,
                agent_timeout_seconds=10,
            ).execute()

        self.assertEqual(result.terminal_reason, "agent_failed")
        self.assertEqual(
            recorder.events[-1],
            (
                "agent_start_failed",
                {"error_type": "FileNotFoundError", "errno": 2},
            ),
        )

    def test_finished_candidates_are_selected_only_after_agent_cleanup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            submission = SubmissionResult(
                submission_id="submission-000001",
                program=program,
                episodes_used=1,
                episodes_remaining=0,
                feedback=Feedback(score=1.0),
                episodes=(
                    EpisodeSummary(
                        status="completed",
                        reward=1.0,
                        steps=1,
                    ),
                ),
            )
            session = FakeSession(
                submissions=(submission,),
                candidate_submission_ids=(submission.submission_id,),
            )
            gateway = FakeGateway()
            selector = FakeCandidateSelector(
                CandidateSelection(
                    submission=submission,
                    validation=None,
                ),
                gateway=gateway,
            )
            recorder = FakeRecorder()

            result = ProgramEvolutionRun(
                benchmark_id="example/benchmark-v1",
                initial_program=program,
                session=session,
                gateway=gateway,
                agent_runner=FakeAgentRunner(
                    AgentExit(
                        returncode=-15,
                        stopped_after_terminal=True,
                    )
                ),
                candidate_selector=selector,
                final_assessor=FakeFinalAssessor(),
                recorder=recorder,
                agent_timeout_seconds=10,
            ).execute()

        self.assertEqual(result.terminal_reason, "finished")
        self.assertEqual(result.final_submission_id, submission.submission_id)
        self.assertEqual(
            result.candidate_submission_ids,
            (submission.submission_id,),
        )
        self.assertEqual(selector.calls, 1)
        self.assertEqual(
            [event for event, _ in recorder.events],
            [
                "agent_started",
                "agent_stopped_after_terminal",
                "final_submission_selected",
                "run_finished",
            ],
        )

    def test_validation_failure_is_terminal_without_a_final_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            submission = SubmissionResult(
                submission_id="submission-000001",
                program=program,
                episodes_used=1,
                episodes_remaining=0,
                feedback=Feedback(score=1.0),
                episodes=(
                    EpisodeSummary(
                        status="completed",
                        reward=1.0,
                        steps=1,
                    ),
                ),
            )
            recorder = FakeRecorder()

            result = ProgramEvolutionRun(
                benchmark_id="example/benchmark-v1",
                initial_program=program,
                session=FakeSession(
                    submissions=(submission,),
                    candidate_submission_ids=(
                        submission.submission_id,
                    ),
                ),
                gateway=FakeGateway(),
                agent_runner=FakeAgentRunner(AgentExit(returncode=0)),
                candidate_selector=FakeCandidateSelector(fail=True),
                final_assessor=FakeFinalAssessor(),
                recorder=recorder,
                agent_timeout_seconds=10,
            ).execute()

        self.assertEqual(result.terminal_reason, "validation_failed")
        self.assertIsNone(result.final_program)
        self.assertIsNone(result.final_submission_id)
        self.assertEqual(
            result.candidate_submission_ids,
            (submission.submission_id,),
        )
        self.assertIsNone(result.validation)
        self.assertEqual(recorder.events[-1][0], "validation_failed")

    def test_assessment_failure_retains_the_selected_final_program(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            submission = SubmissionResult(
                submission_id="submission-000001",
                program=program,
                episodes_used=1,
                episodes_remaining=0,
                feedback=Feedback(score=1.0),
                episodes=(
                    EpisodeSummary(
                        status="completed",
                        reward=1.0,
                        steps=1,
                    ),
                ),
            )
            gateway = FakeGateway()
            assessor = FakeFinalAssessor(
                gateway=gateway,
                fail=True,
            )
            recorder = FakeRecorder()

            result = ProgramEvolutionRun(
                benchmark_id="example/benchmark-v1",
                initial_program=program,
                session=FakeSession(
                    submissions=(submission,),
                    candidate_submission_ids=(
                        submission.submission_id,
                    ),
                ),
                gateway=gateway,
                agent_runner=FakeAgentRunner(AgentExit(returncode=0)),
                candidate_selector=FakeCandidateSelector(
                    CandidateSelection(
                        submission=submission,
                        validation=None,
                    )
                ),
                final_assessor=assessor,
                recorder=recorder,
                agent_timeout_seconds=10,
            ).execute()

        self.assertEqual(result.terminal_reason, "assessment_failed")
        self.assertEqual(result.final_program, program)
        self.assertEqual(
            result.final_submission_id,
            submission.submission_id,
        )
        self.assertIsNone(result.assessment)
        self.assertEqual(assessor.calls, [submission.submission_id])
        self.assertEqual(
            [name for name, _ in recorder.events][-2:],
            ["final_submission_selected", "assessment_failed"],
        )


if __name__ == "__main__":
    unittest.main()
