from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

from evopolicygym.execution.process.agent.runner import (
    ProcessAgentRunner,
)
from evopolicygym.run._agent import AgentOutcome, TerminalSignal
from evopolicygym.run._session.runner import SessionDispatchingAgentRunner


class NeverTerminal:
    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        return False


class ImmediateTerminal:
    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        return True


def runner(root: Path, source: str) -> ProcessAgentRunner:
    return ProcessAgentRunner(
        command=(sys.executable, "-c", source),
        workspace=root,
        environment={},
        stdout_path=root / "stdout.log",
        stderr_path=root / "stderr.log",
    )


class AgentOutcomeTests(unittest.TestCase):
    def test_exit_classifications_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            AgentOutcome(
                returncode=-15,
                timed_out=True,
                stopped_after_terminal=True,
            )


class ProcessAgentRunnerTests(unittest.TestCase):
    def test_session_requests_dispatch_on_the_calling_thread(self) -> None:
        release_agent = threading.Event()
        dispatcher_threads: list[threading.Thread] = []
        agent_threads: list[threading.Thread] = []

        class WaitingAgentRunner:
            def run(
                self,
                terminal: TerminalSignal,
                *,
                timeout_seconds: float,
            ) -> AgentOutcome:
                del terminal, timeout_seconds
                agent_threads.append(threading.current_thread())
                if not release_agent.wait(timeout=1.0):
                    raise AssertionError("Session request was not dispatched")
                return AgentOutcome(returncode=0)

        class ReleasingDispatcher:
            def dispatch_next(self, *, timeout_seconds: float) -> bool:
                del timeout_seconds
                dispatcher_threads.append(threading.current_thread())
                release_agent.set()
                return True

        caller = threading.current_thread()
        outcome = SessionDispatchingAgentRunner(
            agent_runner=WaitingAgentRunner(),
            dispatcher=ReleasingDispatcher(),
        ).run(
            NeverTerminal(),
            timeout_seconds=2.0,
        )

        self.assertEqual(outcome.returncode, 0)
        self.assertTrue(dispatcher_threads)
        self.assertTrue(all(thread is caller for thread in dispatcher_threads))
        self.assertEqual(len(agent_threads), 1)
        self.assertIsNot(agent_threads[0], caller)

    def test_natural_exit_is_not_a_host_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outcome = runner(
                Path(temporary),
                "print('complete')",
            ).run(
                NeverTerminal(),
                timeout_seconds=2,
            )

        self.assertEqual(outcome.returncode, 0)
        self.assertFalse(outcome.timed_out)
        self.assertFalse(outcome.stopped_after_terminal)
        self.assertFalse(outcome.start_failed)

    def test_live_agent_is_classified_when_host_stops_after_terminal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outcome = runner(
                Path(temporary),
                "import time; time.sleep(30)",
            ).run(
                ImmediateTerminal(),
                timeout_seconds=2,
            )

        self.assertIsNotNone(outcome.returncode)
        self.assertNotEqual(outcome.returncode, 0)
        self.assertFalse(outcome.timed_out)
        self.assertTrue(outcome.stopped_after_terminal)
        self.assertFalse(outcome.start_failed)

    def test_timeout_remains_distinct_from_terminal_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outcome = runner(
                Path(temporary),
                "import time; time.sleep(30)",
            ).run(
                NeverTerminal(),
                timeout_seconds=0.05,
            )

        self.assertIsNotNone(outcome.returncode)
        self.assertNotEqual(outcome.returncode, 0)
        self.assertTrue(outcome.timed_out)
        self.assertFalse(outcome.stopped_after_terminal)
        self.assertFalse(outcome.start_failed)


if __name__ == "__main__":
    unittest.main()
