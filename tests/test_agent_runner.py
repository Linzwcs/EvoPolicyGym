from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from evopolicygym.execution.process.agent.runner import (
    AgentExit,
    ProcessAgentRunner,
)


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


class AgentExitTests(unittest.TestCase):
    def test_exit_classifications_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            AgentExit(
                returncode=-15,
                timed_out=True,
                stopped_after_terminal=True,
            )


class ProcessAgentRunnerTests(unittest.TestCase):
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
