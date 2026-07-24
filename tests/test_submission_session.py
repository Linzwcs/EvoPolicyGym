from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from evopolicygym.authoring import (
    Benchmark,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
)
from evopolicygym.errors import EvaluationError, ProgramSourceError
from evopolicygym.evaluation import EvaluationConfig
from evopolicygym.program import Program
from evopolicygym.results import (
    EpisodeSummary,
    EvaluationResult,
    Feedback,
    RunResult,
    SubmissionResult,
)
from evopolicygym.run import RunConfig
from evopolicygym.run._session import (
    FinishReceipt,
    SessionError,
    SubmissionReceipt,
    SubmissionSession,
)


class StubBenchmark:
    @property
    def spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(
            id="example/session-v1",
            description="Session rule fixture.",
            observation_space=None,
            action_space=None,
            metadata={},
            max_episode_steps=1,
            primary_metric="reward",
            score_direction="maximize",
        )

    def episodes(
        self,
        split: str,
        *,
        seed: int,
        count: int,
    ) -> Sequence[EpisodeSpec]:
        del split
        return tuple(
            EpisodeSpec(environment_seed=seed + index)
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        del episode
        raise AssertionError("the fake evaluator owns evaluation")

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        del episodes
        raise AssertionError("the fake evaluator owns evaluation")


class FakeProgramSource:
    def __init__(self, program: Program, *, fail_once: bool = False) -> None:
        self.program = program
        self.fail_once = fail_once

    def capture(self) -> Program:
        if self.fail_once:
            self.fail_once = False
            raise ProgramSourceError("invalid fixture Program")
        return self.program


class FakeEvaluator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.configs: list[EvaluationConfig] = []

    def evaluate(
        self,
        program: Program,
        benchmark: Benchmark,
        config: EvaluationConfig,
        *,
        episode_completed: (
            Callable[[int, int, EpisodeSummary], None] | None
        ) = None,
    ) -> EvaluationResult:
        del benchmark
        self.configs.append(config)
        if self.fail:
            raise EvaluationError("trusted fixture failure")
        episodes = tuple(
            EpisodeSummary(status="completed", reward=1.0, steps=1)
            for _ in range(config.episodes)
        )
        if episode_completed is not None:
            for index, episode in enumerate(episodes, start=1):
                episode_completed(index, len(episodes), episode)
        return EvaluationResult(
            benchmark_id="example/session-v1",
            program_digest=program.digest,
            feedback=Feedback(
                score=float(config.episodes),
                content="fixture",
            ),
            episodes=episodes,
        )


class FakePublisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.results: list[SubmissionResult] = []

    def commit(self, result: SubmissionResult) -> None:
        if self.fail:
            raise OSError("fixture publication failure")
        self.results.append(result)


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def record_event(
        self,
        event: str,
        fields: Mapping[str, object],
    ) -> None:
        self.events.append((event, dict(fields)))

    def commit(self, result: RunResult, agent_exit: object) -> None:
        del result, agent_exit
        raise AssertionError("SubmissionSession does not commit a Run")


def make_program(root: Path) -> Program:
    source = root / "program"
    source.mkdir()
    (source / "policy.py").write_text(
        "def make_policy(context):\n    return object()\n",
        encoding="utf-8",
    )
    return Program.from_directory(source)


class SubmissionSessionTests(unittest.TestCase):
    def test_agent_can_allocate_the_entire_budget_to_one_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            evaluator = FakeEvaluator()
            session = self._session(
                FakeProgramSource(program),
                evaluator,
                FakePublisher(),
                episode_budget=7,
            )

            submitted = session.submit(7)

        self.assertIsInstance(submitted, SubmissionReceipt)
        assert isinstance(submitted, SubmissionReceipt)
        self.assertEqual(submitted.episodes_used, 7)
        self.assertEqual(submitted.episodes_remaining, 0)
        self.assertEqual(evaluator.configs[0].episodes, 7)

    def test_optional_submission_cap_rejects_only_oversized_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                FakePublisher(),
                episode_budget=7,
                max_episodes_per_submission=3,
            )

            rejected = session.submit(4)
            accepted = session.submit(3)

        self.assertIsInstance(rejected, SessionError)
        assert isinstance(rejected, SessionError)
        self.assertEqual(rejected.code, "episode_limit")
        self.assertIsInstance(accepted, SubmissionReceipt)
        assert isinstance(accepted, SubmissionReceipt)
        self.assertEqual(accepted.episodes_remaining, 4)

    def test_invalid_program_does_not_consume_episode_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            source = FakeProgramSource(program, fail_once=True)
            session = self._session(
                source,
                FakeEvaluator(),
                FakePublisher(),
                episode_budget=3,
            )

            rejected = session.submit(3)
            accepted = session.submit(3)

        self.assertIsInstance(rejected, SessionError)
        assert isinstance(rejected, SessionError)
        self.assertEqual(rejected.code, "program_invalid")
        self.assertIsInstance(accepted, SubmissionReceipt)
        assert isinstance(accepted, SubmissionReceipt)
        self.assertEqual(accepted.episodes_remaining, 0)

    def test_evaluation_failure_consumes_reserved_budget_and_closes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            recorder = FakeRecorder()
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(fail=True),
                FakePublisher(),
                recorder=recorder,
                episode_budget=5,
            )

            failed = session.submit(3)
            closed = session.submit(1)

        self.assertIsInstance(failed, SessionError)
        assert isinstance(failed, SessionError)
        self.assertEqual(failed.code, "evaluation_failed")
        self.assertEqual(session.terminal_reason, "evaluation_failed")
        self.assertIsInstance(closed, SessionError)
        assert isinstance(closed, SessionError)
        self.assertEqual(closed.code, "session_closed")
        self.assertEqual(
            recorder.events[-1][1]["episodes_remaining"],
            2,
        )

    def test_published_submission_can_be_selected_as_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            publisher = FakePublisher()
            recorder = FakeRecorder()
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                publisher,
                recorder=recorder,
                episode_budget=5,
            )

            submitted = session.submit(2)
            assert isinstance(submitted, SubmissionReceipt)
            finished = session.finish(submitted.submission_id)

        self.assertIsInstance(finished, FinishReceipt)
        self.assertEqual(session.terminal_reason, "finished")
        self.assertEqual(session.final_program, program)
        self.assertEqual(len(publisher.results), 1)
        episode_events = [
            fields
            for name, fields in recorder.events
            if name == "episode_completed"
        ]
        self.assertEqual(
            episode_events,
            [
                {
                    "submission_id": "submission-000001",
                    "completed": 1,
                    "total": 2,
                    "status": "completed",
                },
                {
                    "submission_id": "submission-000001",
                    "completed": 2,
                    "total": 2,
                    "status": "completed",
                },
            ],
        )

    def test_publication_failure_is_terminal_and_not_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                FakePublisher(fail=True),
            )

            outcome = session.submit(1)

        self.assertIsInstance(outcome, SessionError)
        assert isinstance(outcome, SessionError)
        self.assertEqual(outcome.code, "publication_failed")
        self.assertEqual(session.terminal_reason, "evaluation_failed")
        self.assertEqual(session.submissions, ())

    def test_finish_rejects_unknown_submission_without_closing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                FakePublisher(),
            )

            outcome = session.finish("submission-999999")

        self.assertIsInstance(outcome, SessionError)
        assert isinstance(outcome, SessionError)
        self.assertEqual(outcome.code, "unknown_submission")
        self.assertIsNone(session.terminal_reason)

    def _session(
        self,
        source: FakeProgramSource,
        evaluator: FakeEvaluator,
        publisher: FakePublisher,
        *,
        recorder: FakeRecorder | None = None,
        episode_budget: int = 5,
        max_episodes_per_submission: int | None = None,
    ) -> SubmissionSession:
        return SubmissionSession(
            programs=source,
            evaluator=evaluator,
            publisher=publisher,
            benchmark=StubBenchmark(),
            config=RunConfig(
                episode_budget=episode_budget,
                max_episodes_per_submission=max_episodes_per_submission,
            ),
            recorder=FakeRecorder() if recorder is None else recorder,
        )


if __name__ == "__main__":
    unittest.main()
