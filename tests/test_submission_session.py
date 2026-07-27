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
from evopolicygym.evaluation._plan import PlannedEpisode
from evopolicygym.program import Program
from evopolicygym.results import (
    EpisodeSummary,
    EvaluationResult,
    Feedback,
    RunResult,
    SubmissionResult,
)
from evopolicygym.run import RunConfig, ValidationConfig
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
    def __init__(
        self,
        *,
        fail: bool = False,
        mismatch_environment: bool = False,
    ) -> None:
        self.fail = fail
        self.mismatch_environment = mismatch_environment
        self.plans: list[tuple[PlannedEpisode, ...]] = []

    def evaluate_episodes(
        self,
        program: Program,
        benchmark: Benchmark,
        episodes: tuple[PlannedEpisode, ...],
        *,
        episode_timeout_seconds: float,
        episode_completed: (
            Callable[[int, int, EpisodeSummary], None] | None
        ) = None,
    ) -> EvaluationResult:
        del episode_timeout_seconds
        self.plans.append(episodes)
        if self.fail:
            raise EvaluationError("trusted fixture failure")
        summaries = tuple(
            EpisodeSummary(status="completed", reward=1.0, steps=1)
            for _ in episodes
        )
        if episode_completed is not None:
            for index, summary in enumerate(summaries, start=1):
                episode_completed(index, len(summaries), summary)
        return EvaluationResult(
            benchmark_id="example/session-v1",
            environment_digest=(
                "sha256:" + "0" * 64
                if self.mismatch_environment
                else benchmark.spec.environment_digest
            ),
            program_digest=program.digest,
            feedback=Feedback(
                score=float(len(episodes)),
                content="fixture",
            ),
            episodes=summaries,
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

            submitted = session.submit(list(range(7)))

        self.assertIsInstance(submitted, SubmissionReceipt)
        assert isinstance(submitted, SubmissionReceipt)
        self.assertEqual(submitted.episodes_used, 7)
        self.assertEqual(submitted.episode_indices, tuple(range(7)))
        self.assertEqual(submitted.episodes_remaining, 0)
        self.assertEqual(len(evaluator.plans[0]), 7)

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

            rejected = session.submit(list(range(4)))
            accepted = session.submit(list(range(3)))

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

            rejected = session.submit(list(range(3)))
            accepted = session.submit(list(range(3)))

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

            failed = session.submit(list(range(3)))
            closed = session.submit([0])

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

            submitted = session.submit([0, 1])
            assert isinstance(submitted, SubmissionReceipt)
            finished = session.finish([submitted.submission_id])

        self.assertIsInstance(finished, FinishReceipt)
        assert isinstance(finished, FinishReceipt)
        self.assertEqual(
            finished.candidate_submission_ids,
            (submitted.submission_id,),
        )
        self.assertIsNone(session.terminal_reason)
        self.assertEqual(
            session.candidate_submission_ids,
            (submitted.submission_id,),
        )
        self.assertTrue(session.agent_authority_closed)
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
                    "episode_index": 0,
                    "status": "completed",
                },
                {
                    "submission_id": "submission-000001",
                    "completed": 2,
                    "total": 2,
                    "episode_index": 1,
                    "status": "completed",
                },
            ],
        )

    def test_mismatched_environment_identity_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            publisher = FakePublisher()
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(mismatch_environment=True),
                publisher,
            )

            outcome = session.submit([0])

        self.assertIsInstance(outcome, SessionError)
        assert isinstance(outcome, SessionError)
        self.assertEqual(outcome.code, "evaluation_failed")
        self.assertEqual(session.terminal_reason, "evaluation_failed")
        self.assertEqual(session.submissions, ())
        self.assertEqual(publisher.results, [])

    def test_publication_failure_is_terminal_and_not_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                FakePublisher(fail=True),
            )

            outcome = session.submit([0])

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

            outcome = session.finish(["submission-999999"])

        self.assertIsInstance(outcome, SessionError)
        assert isinstance(outcome, SessionError)
        self.assertEqual(outcome.code, "unknown_submission")
        self.assertIsNone(session.terminal_reason)
        self.assertFalse(session.agent_authority_closed)

    def test_finish_rejections_are_atomic_and_a_valid_retry_closes_authority(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            recorder = FakeRecorder()
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                FakePublisher(),
                recorder=recorder,
                episode_budget=4,
                validation=ValidationConfig(
                    episodes_per_candidate=3,
                    max_candidates=2,
                ),
            )
            first = session.submit([0])
            second = session.submit([0])
            assert isinstance(first, SubmissionReceipt)
            assert isinstance(second, SubmissionReceipt)

            malformed = session.finish([])
            over_limit = session.finish(
                [
                    first.submission_id,
                    second.submission_id,
                    "submission-999999",
                ]
            )
            duplicate = session.finish(
                [first.submission_id, first.submission_id]
            )
            unknown = session.finish(
                [first.submission_id, "submission-999999"]
            )
            accepted = session.finish(
                [second.submission_id, first.submission_id]
            )
            closed = session.submit([0])

        for outcome, code in (
            (malformed, "invalid_request"),
            (over_limit, "candidate_limit"),
            (duplicate, "duplicate_submission"),
            (unknown, "unknown_submission"),
        ):
            self.assertIsInstance(outcome, SessionError)
            assert isinstance(outcome, SessionError)
            self.assertEqual(outcome.code, code)
        self.assertIsInstance(accepted, FinishReceipt)
        self.assertEqual(
            session.candidate_submission_ids,
            (second.submission_id, first.submission_id),
        )
        self.assertIsInstance(closed, SessionError)
        assert isinstance(closed, SessionError)
        self.assertEqual(closed.code, "session_closed")
        self.assertEqual(
            [name for name, _ in recorder.events].count("finish_rejected"),
            4,
        )

    def test_finish_without_validation_accepts_only_one_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            session = self._session(
                FakeProgramSource(program),
                FakeEvaluator(),
                FakePublisher(),
            )
            first = session.submit([0])
            second = session.submit([0])
            assert isinstance(first, SubmissionReceipt)
            assert isinstance(second, SubmissionReceipt)

            outcome = session.finish(
                [first.submission_id, second.submission_id]
            )

        self.assertIsInstance(outcome, SessionError)
        assert isinstance(outcome, SessionError)
        self.assertEqual(outcome.code, "candidate_limit")
        self.assertFalse(session.agent_authority_closed)

    def test_arbitrary_indices_select_the_exact_fixed_pool_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            evaluator = FakeEvaluator()
            session = self._session(
                FakeProgramSource(program),
                evaluator,
                FakePublisher(),
                episode_budget=12,
            )

            first = session.submit([0, 1, 4, 5, 6, 7])
            second = session.submit([1, 4])

        self.assertIsInstance(first, SubmissionReceipt)
        self.assertIsInstance(second, SubmissionReceipt)
        self.assertEqual(
            tuple(item.episode.environment_seed for item in evaluator.plans[0]),
            (0, 1, 4, 5, 6, 7),
        )
        self.assertEqual(evaluator.plans[0][1], evaluator.plans[1][0])
        self.assertEqual(evaluator.plans[0][2], evaluator.plans[1][1])

    def test_invalid_index_sets_do_not_capture_or_consume_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            program = make_program(Path(temporary))
            source = FakeProgramSource(program)
            evaluator = FakeEvaluator()
            session = self._session(
                source,
                evaluator,
                FakePublisher(),
                episode_budget=5,
            )

            outcomes = (
                session.submit([]),
                session.submit([0, 0]),
                session.submit([2, 1]),
                session.submit([5]),
                session.submit([True]),
            )
            accepted = session.submit([0, 2, 4])

        for outcome in outcomes:
            self.assertIsInstance(outcome, SessionError)
            assert isinstance(outcome, SessionError)
            self.assertEqual(outcome.code, "invalid_request")
        self.assertIsInstance(accepted, SubmissionReceipt)
        assert isinstance(accepted, SubmissionReceipt)
        self.assertEqual(accepted.episodes_remaining, 2)
        self.assertEqual(len(evaluator.plans), 1)

    def _session(
        self,
        source: FakeProgramSource,
        evaluator: FakeEvaluator,
        publisher: FakePublisher,
        *,
        recorder: FakeRecorder | None = None,
        episode_budget: int = 5,
        max_episodes_per_submission: int | None = None,
        validation: ValidationConfig | None = None,
    ) -> SubmissionSession:
        benchmark = StubBenchmark()
        config = RunConfig(
            episode_budget=episode_budget,
            max_episodes_per_submission=max_episodes_per_submission,
            validation=validation,
        )
        pool_size = config.episode_pool_size
        assert pool_size is not None
        episode_pool = tuple(
            PlannedEpisode(
                EpisodeSpec(environment_seed=index),
                policy_seed=10_000 + index,
            )
            for index in range(pool_size)
        )
        return SubmissionSession(
            programs=source,
            evaluator=evaluator,
            publisher=publisher,
            benchmark=benchmark,
            spec=benchmark.spec,
            config=config,
            recorder=FakeRecorder() if recorder is None else recorder,
            episode_pool=episode_pool,
        )


if __name__ == "__main__":
    unittest.main()
