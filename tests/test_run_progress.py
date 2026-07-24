from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym.program import Program
from evopolicygym.run import ConsoleProgress, RunConfig, RunEvent
from evopolicygym.run._directory import (
    RunDirectoryPaths,
    RunDirectoryRecorder,
)


class FailingObserver:
    def __init__(self) -> None:
        self.calls = 0

    def on_event(self, event: RunEvent, /) -> None:
        del event
        self.calls += 1
        raise RuntimeError("display failed")


def make_program(root: Path) -> Program:
    source = root / "program"
    source.mkdir()
    (source / "policy.py").write_text(
        "def make_policy(context):\n    return object()\n",
        encoding="utf-8",
    )
    return Program.from_directory(source)


def event(
    name: str,
    monotonic_ns: int,
    **fields: str | int | float | bool | None,
) -> RunEvent:
    return RunEvent(
        name=name,
        time_unix_ns=1,
        monotonic_ns=monotonic_ns,
        fields=fields,
    )


class RunEventTests(unittest.TestCase):
    def test_event_detaches_and_freezes_scalar_fields(self) -> None:
        fields: dict[str, str | int | float | bool | None] = {
            "submission_id": "submission-000001",
            "completed": 1,
        }
        published = RunEvent(
            name="episode_completed",
            time_unix_ns=1,
            monotonic_ns=2,
            fields=fields,
        )
        fields["completed"] = 2

        self.assertEqual(published.fields["completed"], 1)
        with self.assertRaises(TypeError):
            published.fields["completed"] = 3  # type: ignore[index]

    def test_event_rejects_reserved_and_non_scalar_fields(self) -> None:
        with self.assertRaises(ValueError):
            RunEvent(
                name="example",
                time_unix_ns=1,
                monotonic_ns=2,
                fields={"event": "nested"},
            )
        with self.assertRaises(TypeError):
            RunEvent(
                name="example",
                time_unix_ns=1,
                monotonic_ns=2,
                fields={"items": []},  # type: ignore[dict-item]
            )


class RunDirectoryObserverTests(unittest.TestCase):
    def test_observer_failure_does_not_change_persisted_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observer = FailingObserver()
            with RunDirectoryRecorder(
                paths=RunDirectoryPaths.under(root),
                benchmark_id="example/progress-v1",
                initial_program=make_program(root),
                config=RunConfig(),
                agent_identity={},
                observer=observer,
            ) as recorder:
                recorder.record_event("first", {"value": 1})
                recorder.record_event("second", {"value": 2})
            documents = tuple(
                json.loads(line)
                for line in (root / "events.jsonl").read_text().splitlines()
            )

        self.assertEqual(observer.calls, 1)
        self.assertEqual(
            [document["event"] for document in documents],
            ["first", "second"],
        )


class ConsoleProgressTests(unittest.TestCase):
    def test_plain_progress_renders_operator_lifecycle(self) -> None:
        output = io.StringIO()
        progress = ConsoleProgress(output, mode="plain")

        progress.on_event(
            event(
                "agent_started",
                1_000_000_000,
                benchmark_id="cartpole/v1",
            )
        )
        progress.on_event(
            event(
                "evaluation_started",
                2_000_000_000,
                submission_id="submission-000001",
                episodes=3,
                episodes_remaining=3,
            )
        )
        progress.on_event(
            event(
                "episode_completed",
                2_500_000_000,
                submission_id="submission-000001",
                completed=1,
                total=3,
                status="completed",
            )
        )
        progress.on_event(
            event(
                "submission_published",
                3_000_000_000,
                submission_id="submission-000001",
                score=500.0,
                episodes_remaining=3,
            )
        )
        progress.on_event(
            event(
                "run_finished",
                4_000_000_000,
                submission_id="submission-000001",
            )
        )

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "Agent started · cartpole/v1",
                (
                    "submission-000001 · evaluating 3 Episodes"
                    " · budget 3 remaining"
                ),
                (
                    "submission-000001 · Episodes 1/3"
                    " · 0.5s · completed"
                ),
                "submission-000001 · score 500 · budget 3 remaining",
                "Run finished · selected submission-000001",
            ],
        )

    def test_unknown_future_event_is_ignored(self) -> None:
        output = io.StringIO()
        ConsoleProgress(output, mode="plain").on_event(
            event("future_event", 1)
        )
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
