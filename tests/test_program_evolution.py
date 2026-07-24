from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from evopolicygym.execution.process.agent.runner import AgentExit
from evopolicygym.program import Program
from evopolicygym.results import RunResult, RunTerminalReason, SubmissionResult
from evopolicygym.run._service import (
    ProgramEvolutionRun,
    TerminalSignal,
)


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
    ) -> None:
        self.submissions: tuple[SubmissionResult, ...] = ()
        self.final_submission_id: str | None = None
        self.final_program: Program | None = None
        self.terminal_reason = terminal_reason
        self.authority_exhausted = authority_exhausted


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


if __name__ == "__main__":
    unittest.main()
