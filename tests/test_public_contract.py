from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import evopolicygym
from evopolicygym import (
    Benchmark,
    EvaluationConfig,
    EvaluationResult,
    Program,
    RunConfig,
    RunResult,
    evaluate,
    run,
)
from evopolicygym.agents import Codex, command_invocation
from evopolicygym.artifacts import (
    FEEDBACK_MAX_ARTIFACTS,
    Artifact,
)
from evopolicygym.authoring import (
    BenchmarkFixture,
    BenchmarkSpec,
    Environment,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Step,
    Transition,
    check_benchmark,
)
from evopolicygym.errors import EvaluationError
from evopolicygym.execution import ProcessExecution
from evopolicygym.execution.process.agent.runner import (
    build_agent_environment,
)
from evopolicygym.policy import (
    Policy,
    PolicyContext,
    PolicyValue,
    TensorValue,
)
from evopolicygym.results import (
    EpisodeSummary,
    Feedback,
    SubmissionResult,
)
from evopolicygym.run import RunEvent
from evopolicygym.run._feedback import record_submission
from evopolicygym.run._service import run_process_agent


class RecordingRunObserver:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def on_event(self, event: RunEvent, /) -> None:
        self.events.append(event)


class ConstantPolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        return observation


class CounterEnvironment:
    def __init__(self, episode: EpisodeSpec) -> None:
        self._episode = episode
        self._position = 0

    def reset(self) -> PolicyValue:
        self._position = int(self._episode.environment_seed % 3)
        return self._position

    def step(self, action: PolicyValue) -> Step:
        if type(action) is not int or action not in {-1, 1}:
            raise ValueError("invalid Action")
        self._position += action
        return Step(
            observation=self._position,
            reward=float(self._position),
            terminated=True,
            metrics={"private_position": self._position},
        )

    def close(self) -> None:
        pass


class CounterBenchmark:
    @property
    def spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(
            id="example/counter-v1",
            description="A deterministic counter.",
            observation_space={"type": "integer"},
            action_space={"enum": [-1, 1]},
            metadata={"family": "test"},
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
            EpisodeSpec(
                environment_seed=seed + index,
                scenario={"target": index},
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        return CounterEnvironment(episode)

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        score = sum(episode.total_reward for episode in episodes)
        return Feedback(
            score=score,
            content={
                "message": "Evaluation complete.",
                "episodes": len(episodes),
            },
            artifacts=(
                Artifact(
                    name="summary.txt",
                    media_type="text/plain",
                    content=f"{len(episodes)} episodes".encode(),
                ),
            ),
        )


def require_benchmark(benchmark: Benchmark) -> Benchmark:
    return benchmark


class PublicContractTests(unittest.TestCase):
    def test_root_exports_only_common_user_workflow_values(self) -> None:
        self.assertEqual(
            set(evopolicygym.__all__),
            {
                "Benchmark",
                "EvaluationConfig",
                "EvaluationResult",
                "Program",
                "RunConfig",
                "RunResult",
                "__version__",
                "evaluate",
                "run",
            },
        )
        self.assertEqual(evopolicygym.__version__, "0.3.0")
        self.assertFalse(hasattr(evopolicygym, "Artifact"))
        self.assertFalse(hasattr(evopolicygym, "PolicyValue"))
        self.assertTrue(callable(evaluate))
        self.assertTrue(callable(run))

    def test_policy_and_benchmark_are_structural(self) -> None:
        policy = ConstantPolicy()
        benchmark = require_benchmark(CounterBenchmark())
        episode = benchmark.episodes("train", seed=7, count=1)[0]

        self.assertIsInstance(policy, Policy)
        self.assertIsInstance(benchmark, Benchmark)
        self.assertIsInstance(benchmark.make_environment(episode), Environment)

    def test_policy_context_detaches_mutable_values(self) -> None:
        observation_space: PolicyValue = {"shape": [4]}
        metadata: dict[str, PolicyValue] = {"name": "counter"}
        context = PolicyContext(
            observation_space=observation_space,
            action_space={"enum": [-1, 1]},
            metadata=metadata,
            policy_seed=9,
        )

        assert isinstance(observation_space, dict)
        shape = observation_space["shape"]
        assert isinstance(shape, list)
        shape.append(5)
        metadata["name"] = "changed"

        self.assertEqual(context.observation_space, {"shape": [4]})
        self.assertEqual(context.metadata["name"], "counter")

    def test_tensor_value_validates_shape_and_finite_data(self) -> None:
        tensor = TensorValue(
            dtype="float32",
            shape=(2,),
            data=b"\x00\x00\x80?\x00\x00\x00@",
        )

        self.assertEqual(tensor.shape, (2,))
        with self.assertRaises(ValueError):
            TensorValue(dtype="float32", shape=(2,), data=b"short")

    def test_episode_record_keeps_private_input_out_of_public_summary(self) -> None:
        episode = EpisodeSpec(
            environment_seed=71,
            scenario={"private_case": "A"},
        )
        step = Step(
            observation=1,
            reward=2.5,
            terminated=True,
            metrics={"private_measurement": 9},
        )
        record = EpisodeRecord(
            episode=episode,
            policy_seed=99,
            initial_observation=0,
            transitions=(Transition(action=1, step=step),),
        )
        summary = EpisodeSummary(
            status="completed",
            reward=record.total_reward,
            steps=record.steps,
        )

        self.assertEqual(record.episode.environment_seed, 71)
        self.assertEqual(summary.reward, 2.5)
        self.assertFalse(hasattr(summary, "episode"))
        self.assertFalse(hasattr(summary, "policy_seed"))

    def test_artifacts_reject_host_or_parent_paths(self) -> None:
        for name in ("", "/tmp/report.txt", "../report.txt", "a/../report.txt", r"a\report"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Artifact(name=name, media_type="text/plain", content=b"unsafe")
        artifacts = tuple(
            Artifact(
                name=f"reports/{index}.txt",
                media_type="text/plain",
                content=b"",
            )
            for index in range(FEEDBACK_MAX_ARTIFACTS + 1)
        )
        with self.assertRaises(ValueError):
            Feedback(
                score=0.0,
                content="too many files",
                artifacts=artifacts,
            )

    def test_feedback_accepts_detached_benchmark_defined_content(self) -> None:
        content: dict[str, PolicyValue] = {
            "kind": "benchmark-specific",
            "message": "before",
            "diagnostics": [1, 2, 3],
        }
        feedback = Feedback(score=2.5, content=content)
        content["message"] = "after"

        self.assertEqual(
            feedback.content,
            {
                "kind": "benchmark-specific",
                "message": "before",
                "diagnostics": [1, 2, 3],
            },
        )
        with self.assertRaises(TypeError):
            Feedback(score=1.0, content=Path("/private/host/path"))  # type: ignore[arg-type]

    def test_configs_are_finite_and_bounded(self) -> None:
        evaluation = EvaluationConfig(episodes=2, seed=3)
        run = RunConfig(
            max_submissions=4,
            episode_budget=20,
        )
        agent = Codex(model="gpt-5")

        self.assertEqual(evaluation.episodes, 2)
        self.assertEqual(run.max_submissions, 4)
        self.assertEqual(run.episode_budget, 20)
        self.assertIsNone(run.max_episodes_per_submission)
        capped = RunConfig(
            episode_budget=20,
            max_episodes_per_submission=5,
        )
        self.assertEqual(capped.max_episodes_per_submission, 5)
        self.assertEqual(agent.model, "gpt-5")
        with self.assertRaises(ValueError):
            RunConfig(episode_budget=0)
        with self.assertRaises(ValueError):
            RunConfig(
                episode_budget=2,
                max_episodes_per_submission=3,
            )

    def test_public_results_compose_without_private_episode_records(self) -> None:
        feedback = Feedback(score=1.0, content={"outcome": "passed"})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "policy"
            directory.mkdir()
            (directory / "policy.py").write_text(
                "def make_policy(context):\n    return object()\n",
                encoding="utf-8",
            )
            program = Program.from_directory(directory)
        evaluation = EvaluationResult(
            benchmark_id="example/counter-v1",
            program_digest=program.digest,
            feedback=feedback,
            episodes=(
                EpisodeSummary(status="completed", reward=1.0, steps=1),
            ),
        )
        submission = SubmissionResult(
            submission_id="submission-1",
            program=program,
            episodes_used=1,
            episodes_remaining=2,
            feedback=feedback,
            episodes=evaluation.episodes,
        )

        self.assertEqual(evaluation.feedback.score, 1.0)
        self.assertEqual(submission.episodes_remaining, 2)


class ProgramTests(unittest.TestCase):
    def test_program_is_a_detached_reproducible_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "policy.py").write_text(
                "def make_policy(context):\n    return object()\n",
                encoding="utf-8",
            )
            (source / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

            first = Program.from_directory(source)
            second = Program.from_directory(source)
            (source / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")

            self.assertEqual(first.digest, second.digest)
            self.assertEqual(first.read_bytes("helper.py"), b"VALUE = 1\n")
            self.assertEqual(first.entrypoint, "policy.py:make_policy")
            self.assertNotIn(str(source), repr(first))

            submission = SubmissionResult(
                submission_id="submission-1",
                program=first,
                episodes_used=1,
                episodes_remaining=0,
                feedback=Feedback(score=1.0, content="complete"),
                episodes=(
                    EpisodeSummary(status="completed", reward=1.0, steps=1),
                ),
            )
            run = RunResult(
                final_program=first,
                final_submission_id=submission.submission_id,
                submissions=(submission,),
                terminal_reason="finished",
            )
            self.assertEqual(run.final_program, first)

            materialized = root / "materialized"
            first.write_to(materialized)
            self.assertEqual(
                (materialized / "helper.py").read_bytes(),
                b"VALUE = 1\n",
            )

    def test_benchmark_artifact_metadata_and_content_are_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "policy.py").write_text(
                "def make_policy(context):\n    return object()\n",
                encoding="utf-8",
            )
            program = Program.from_directory(source)
            trace = Artifact(
                name="traces/episodes.jsonl",
                media_type="application/x-ndjson",
                content=b'{"type":"episode"}\n',
            )
            submission = SubmissionResult(
                submission_id="submission-000001",
                program=program,
                episodes_used=1,
                episodes_remaining=0,
                feedback=Feedback(
                    score=1.0,
                    content={
                        "kind": "custom-diagnostic",
                        "nested": {
                            "message": "complete",
                            "values": [1, 2, 3],
                        },
                    },
                    artifacts=(trace,),
                ),
                episodes=(
                    EpisodeSummary(
                        status="completed",
                        reward=1.0,
                        steps=1,
                    ),
                ),
            )

            record_submission(root / "submissions", submission)
            submission_root = (
                root / "submissions" / "submission-000001"
            )
            document = json.loads(
                (submission_root / "feedback.json").read_text()
            )
            feedback_content = document["content"]
            metadata = document["artifacts"][0]
            artifact_content = (
                submission_root / metadata["path"]
            ).read_bytes()

        self.assertEqual(
            feedback_content,
            {
                "kind": "custom-diagnostic",
                "nested": {
                    "message": "complete",
                    "values": [1, 2, 3],
                },
            },
        )
        self.assertNotIn("summary", document)
        self.assertNotIn("metrics", document)
        self.assertEqual(metadata["name"], "traces/episodes.jsonl")
        self.assertEqual(metadata["media_type"], "application/x-ndjson")
        self.assertNotIn("role", metadata)
        self.assertNotIn("schema", metadata)
        self.assertEqual(artifact_content, trace.read_bytes())

    def test_program_requires_policy_entrypoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                Program.from_directory(root)


class RecordingEnvironment:
    def __init__(self, *, fail: bool = False) -> None:
        self.actions: list[PolicyValue] = []
        self.closed = False
        self.fail = fail

    def reset(self) -> PolicyValue:
        return 0

    def step(self, action: PolicyValue) -> Step:
        self.actions.append(action)
        if self.fail:
            raise RuntimeError("private Environment failure")
        expected = len(self.actions)
        if type(action) is not int or action != expected:
            raise InvalidAction()
        return Step(
            observation=expected,
            reward=1.0,
            terminated=expected == 2,
            metrics={"private_action": action},
        )

    def close(self) -> None:
        self.closed = True


class RecordingBenchmark:
    def __init__(self, *, environment_failure: bool = False) -> None:
        self.environments: list[RecordingEnvironment] = []
        self.environment_failure = environment_failure

    @property
    def spec(self) -> BenchmarkSpec:
        return BenchmarkSpec(
            id="example/recording-v1",
            description="Records Policy Actions.",
            observation_space={"type": "integer"},
            action_space={"type": "integer"},
            metadata={"public": True},
            max_episode_steps=2,
            primary_metric="completed",
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
            EpisodeSpec(
                environment_seed=seed + index,
                scenario={"private_index": index},
            )
            for index in range(count)
        )

    def make_environment(self, episode: EpisodeSpec) -> Environment:
        del episode
        environment = RecordingEnvironment(fail=self.environment_failure)
        self.environments.append(environment)
        return environment

    def feedback(self, episodes: Sequence[EpisodeRecord]) -> Feedback:
        completed = sum(episode.policy_failure is None for episode in episodes)
        return Feedback(
            score=float(completed),
            content={
                "status": "complete",
                "completed": completed,
            },
            artifacts=(
                Artifact(
                    name="summary.txt",
                    media_type="text/plain",
                    content=f"completed={completed}\n".encode(),
                ),
            ),
        )


class InvalidObservationEnvironment(RecordingEnvironment):
    def reset(self) -> PolicyValue:
        return object()  # type: ignore[return-value]


class InvalidObservationBenchmark(RecordingBenchmark):
    def make_environment(self, episode: EpisodeSpec) -> Environment:
        del episode
        environment = InvalidObservationEnvironment()
        self.environments.append(environment)
        return environment


def captured_program(root: Path, source: str) -> Program:
    directory = root / "policy"
    directory.mkdir()
    (directory / "policy.py").write_text(source, encoding="utf-8")
    return Program.from_directory(directory)


class DirectEvaluationTests(unittest.TestCase):
    def test_each_episode_gets_fresh_policy_but_keeps_same_episode_state(self) -> None:
        source = """\
class CounterPolicy:
    def __init__(self):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return self.calls


def make_policy(context):
    return CounterPolicy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = RecordingBenchmark()

            result = evaluate(
                program,
                benchmark,
                execution=ProcessExecution.unsafe(),
                config=EvaluationConfig(episodes=2, seed=7),
            )

        self.assertEqual(result.feedback.score, 2.0)
        self.assertEqual(
            result.feedback.content,
            {
                "status": "complete",
                "completed": 2,
            },
        )
        self.assertEqual(
            [environment.actions for environment in benchmark.environments],
            [[1, 2], [1, 2]],
        )
        self.assertTrue(all(environment.closed for environment in benchmark.environments))
        self.assertEqual(
            tuple(episode.status for episode in result.episodes),
            ("completed", "completed"),
        )

    def test_policy_exception_stops_before_environment_step(self) -> None:
        source = """\
class BrokenPolicy:
    def act(self, observation):
        raise RuntimeError("private Policy failure")


def make_policy(context):
    return BrokenPolicy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = RecordingBenchmark()

            result = evaluate(
                program,
                benchmark,
                execution=ProcessExecution.unsafe(),
            )

        self.assertEqual(result.episodes[0].status, "policy_failed")
        self.assertEqual(result.episodes[0].failure, "exception")
        self.assertEqual(benchmark.environments[0].actions, [])
        self.assertTrue(benchmark.environments[0].closed)

    def test_invalid_action_is_not_repaired_or_retried(self) -> None:
        source = """\
class InvalidPolicy:
    def act(self, observation):
        return 99


def make_policy(context):
    return InvalidPolicy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = RecordingBenchmark()

            result = evaluate(
                program,
                benchmark,
                execution=ProcessExecution.unsafe(),
            )

        self.assertEqual(result.episodes[0].failure, "invalid_action")
        self.assertEqual(benchmark.environments[0].actions, [99])

    def test_environment_fault_aborts_instead_of_becoming_policy_penalty(self) -> None:
        source = """\
class Policy:
    def act(self, observation):
        return 1


def make_policy(context):
    return Policy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = RecordingBenchmark(environment_failure=True)

            with self.assertRaises(EvaluationError):
                evaluate(
                    program,
                    benchmark,
                    execution=ProcessExecution.unsafe(),
                )

        self.assertTrue(benchmark.environments[0].closed)

    def test_invalid_trusted_observation_aborts_instead_of_penalizing_policy(self) -> None:
        source = """\
class Policy:
    def act(self, observation):
        return 1


def make_policy(context):
    return Policy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = InvalidObservationBenchmark()

            with self.assertRaises(EvaluationError):
                evaluate(
                    program,
                    benchmark,
                    execution=ProcessExecution.unsafe(),
                )

        self.assertTrue(benchmark.environments[0].closed)

    def test_non_policy_value_action_is_a_protocol_failure(self) -> None:
        source = """\
class InvalidPolicy:
    def act(self, observation):
        return object()


def make_policy(context):
    return InvalidPolicy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = RecordingBenchmark()

            result = evaluate(
                program,
                benchmark,
                execution=ProcessExecution.unsafe(),
            )

        self.assertEqual(result.episodes[0].failure, "protocol_error")
        self.assertEqual(benchmark.environments[0].actions, [])

    def test_timeout_is_a_policy_failure_and_consumes_the_episode(self) -> None:
        source = """\
import time


class SlowPolicy:
    def act(self, observation):
        time.sleep(5)
        return 1


def make_policy(context):
    return SlowPolicy()
"""
        with tempfile.TemporaryDirectory() as temporary:
            program = captured_program(Path(temporary), source)
            benchmark = RecordingBenchmark()

            result = evaluate(
                program,
                benchmark,
                execution=ProcessExecution.unsafe(),
                config=EvaluationConfig(episode_timeout_seconds=0.1),
            )

        self.assertEqual(result.episodes[0].failure, "timeout")
        self.assertEqual(benchmark.environments[0].actions, [])
        self.assertTrue(benchmark.environments[0].closed)

    def test_process_execution_requires_explicit_unsafe_factory(self) -> None:
        with self.assertRaises(TypeError):
            ProcessExecution()


class AgentSessionTests(unittest.TestCase):
    def test_agent_edits_submits_reads_public_feedback_and_finishes(self) -> None:
        initial_source = """\
class InvalidPolicy:
    def act(self, observation):
        return 99


def make_policy(context):
    return InvalidPolicy()
"""
        improved_source = """\
class CounterPolicy:
    def __init__(self):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return self.calls


def make_policy(context):
    return CounterPolicy()
"""
        unsubmitted_source = """\
def make_policy(context):
    raise RuntimeError("unsubmitted edit")
"""
        agent_source = f"""\
import json
import os
from pathlib import Path
import subprocess
import sys


workspace = Path(os.environ["EVOPOLICYGYM_WORKSPACE"])


def call(*arguments):
    completed = subprocess.run(
        [sys.executable, "-m", "evopolicygym.cli", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


first = call("submit", "program", "--episodes", "1")
latest = json.loads((workspace / "feedback" / "latest.json").read_text())
feedback = json.loads(
    (workspace / "feedback" / latest["feedback"]).read_text()
)
assert first["result"]["submission_id"] == "submission-000001"
assert first["result"]["episodes_remaining"] == 1
assert feedback["episodes"][0]["failure"] == "invalid_action"
assert feedback["content"] == {{"status": "complete", "completed": 0}}
assert not (workspace / "events.jsonl").exists()
assert not (workspace / "agent").exists()

(workspace / "program" / "policy.py").write_text(
    {improved_source!r},
    encoding="utf-8",
)
second = call("submit", "program", "--episodes", "1")
latest = json.loads((workspace / "feedback" / "latest.json").read_text())
feedback = json.loads(
    (workspace / "feedback" / latest["feedback"]).read_text()
)
artifact = workspace / "feedback" / "submissions" / (
    second["result"]["submission_id"]
) / feedback["artifacts"][0]["path"]
assert feedback["score"] == 1.0
assert feedback["content"] == {{"status": "complete", "completed": 1}}
assert artifact.read_text() == "completed=1\\n"
(workspace / "program" / "policy.py").write_text(
    {unsubmitted_source!r},
    encoding="utf-8",
)
call("finish", second["result"]["submission_id"])
print("fake-agent-finished")
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = captured_program(root, initial_source)
            agent_script = root / "fake_agent.py"
            agent_script.write_text(agent_source, encoding="utf-8")
            run_directory = root / "run-record"
            observer = RecordingRunObserver()

            result = run_process_agent(
                program,
                RecordingBenchmark(),
                invocation=command_invocation(
                    (sys.executable, str(agent_script)),
                ),
                run_directory=run_directory,
                config=RunConfig(
                    max_submissions=2,
                    episode_budget=2,
                    agent_timeout_seconds=10,
                ),
                observer=observer,
            )

            stdout = (run_directory / "agent" / "stdout.log").read_text()
            events = tuple(
                json.loads(line)
                for line in (run_directory / "events.jsonl").read_text().splitlines()
            )
            manifest = json.loads((run_directory / "run.json").read_text())
            retained_workspace = run_directory / "workspace"
            first_record = (
                run_directory
                / "submissions"
                / "submission-000001"
            )
            second_record = (
                run_directory
                / "submissions"
                / "submission-000002"
            )
            first_record_source = (
                first_record / "program" / "policy.py"
            ).read_bytes()
            second_record_source = (
                second_record / "program" / "policy.py"
            ).read_bytes()
            workspace_feedback = (
                retained_workspace
                / "feedback"
                / "submissions"
                / "submission-000002"
                / "feedback.json"
            )
            host_feedback = second_record / "feedback.json"
            feedback_inodes = (
                workspace_feedback.stat().st_ino,
                host_feedback.stat().st_ino,
            )
            initial_source_record = (
                run_directory / "initial" / "program" / "policy.py"
            ).read_bytes()
            retained_workspace_source = (
                retained_workspace / "program" / "policy.py"
            ).read_bytes()
            control_exists = (run_directory / "control").exists()

        self.assertEqual(result.terminal_reason, "finished")
        self.assertEqual(result.final_submission_id, "submission-000002")
        self.assertEqual(len(result.submissions), 2)
        self.assertEqual(result.submissions[0].episodes[0].failure, "invalid_action")
        self.assertEqual(result.submissions[1].feedback.score, 1.0)
        self.assertEqual(result.submissions[1].episodes_remaining, 0)
        self.assertIsNotNone(result.final_program)
        assert result.final_program is not None
        self.assertEqual(
            result.final_program.read_bytes("policy.py"),
            improved_source.encode(),
        )
        self.assertEqual(result.submissions[0].program.read_bytes("policy.py"), initial_source.encode())
        self.assertEqual(result.submissions[1].program.read_bytes("policy.py"), improved_source.encode())
        self.assertIn("fake-agent-finished", stdout)
        self.assertEqual(
            [event["event"] for event in events].count("submission_published"),
            2,
        )
        self.assertEqual(
            [event["event"] for event in events].count("episode_completed"),
            2,
        )
        self.assertEqual(
            [event.name for event in observer.events],
            [event["event"] for event in events],
        )
        self.assertEqual(manifest["schema"], "evopolicygym/run-record/v1")
        self.assertEqual(manifest["terminal_reason"], "finished")
        self.assertEqual(manifest["final_submission_id"], "submission-000002")
        self.assertFalse(manifest["agent"]["stopped_after_terminal"])
        self.assertEqual(first_record_source, initial_source.encode())
        self.assertEqual(second_record_source, improved_source.encode())
        self.assertEqual(initial_source_record, initial_source.encode())
        self.assertEqual(retained_workspace_source, unsubmitted_source.encode())
        self.assertNotEqual(*feedback_inodes)
        self.assertFalse(control_exists)

    def test_trusted_evaluation_failure_consumes_reserved_budget(self) -> None:
        policy_source = """\
class Policy:
    def act(self, observation):
        return 1


def make_policy(context):
    return Policy()
"""
        agent_source = """\
import subprocess
import sys


subprocess.run(
    [
        sys.executable,
        "-m",
        "evopolicygym.cli",
        "submit",
        "program",
        "--episodes",
        "2",
    ],
    check=False,
)
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = captured_program(root, policy_source)
            agent_script = root / "failing_agent.py"
            agent_script.write_text(agent_source, encoding="utf-8")
            run_directory = root / "run-record"

            result = run_process_agent(
                program,
                RecordingBenchmark(environment_failure=True),
                invocation=command_invocation(
                    (sys.executable, str(agent_script)),
                ),
                run_directory=run_directory,
                config=RunConfig(
                    max_submissions=2,
                    episode_budget=3,
                    agent_timeout_seconds=10,
                ),
            )
            events = tuple(
                json.loads(line)
                for line in (run_directory / "events.jsonl").read_text().splitlines()
            )
            manifest = json.loads((run_directory / "run.json").read_text())

        self.assertEqual(result.terminal_reason, "evaluation_failed")
        self.assertEqual(result.submissions, ())
        failure = next(
            event for event in events if event["event"] == "evaluation_failed"
        )
        self.assertEqual(failure["episodes_remaining"], 1)
        self.assertEqual(manifest["terminal_reason"], "evaluation_failed")


class AgentEnvironmentTests(unittest.TestCase):
    def test_cli_path_and_session_address_are_valid_from_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary) / "run"
            workspace = run_directory / "workspace"
            control = run_directory / "control"
            workspace.mkdir(parents=True)
            control.mkdir()

            environment = build_agent_environment(
                control / "session.sock",
                workspace,
                inherited_names=(),
            )

        command_directory = str(
            Path(sys.executable).parent.resolve(strict=True)
        )
        self.assertEqual(
            environment["EVOPOLICYGYM_SESSION_SOCKET"],
            os.path.join("..", "control", "session.sock"),
        )
        self.assertEqual(
            environment["PATH"].split(os.pathsep)[0],
            command_directory,
        )
        self.assertIsNotNone(
            shutil.which("evopolicygym", path=environment["PATH"])
        )


class CodexRunTests(unittest.TestCase):
    def test_codex_runs_from_workspace_and_commits_program(self) -> None:
        initial_source = """\
class InvalidPolicy:
    def act(self, observation):
        return 99


def make_policy(context):
    return InvalidPolicy()
"""
        improved_source = """\
class CounterPolicy:
    def __init__(self):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return self.calls


def make_policy(context):
    return CounterPolicy()
"""
        api_key = "test-secret-never-retained"
        fake_codex_source = f"""#!{sys.executable}
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


arguments = sys.argv[1:]
workspace = Path(os.environ["EVOPOLICYGYM_WORKSPACE"])
assert Path.cwd() == workspace
assert os.environ["EVOPOLICYGYM_SESSION_SOCKET"] == os.path.join(
    "..", "control", "session.sock"
)
assert shutil.which("evopolicygym") is not None
assert arguments[:3] == ["--ask-for-approval", "never", "exec"]
for flag in (
    "--ephemeral",
    "--json",
    "--skip-git-repo-check",
    "--ignore-user-config",
    "--ignore-rules",
):
    assert flag in arguments
assert arguments[arguments.index("--model") + 1] == "fake-model"
assert arguments[arguments.index("--sandbox") + 1] == "danger-full-access"
assert arguments[arguments.index("--color") + 1] == "never"
prompt = arguments[-1]
assert "program/" in prompt
assert "feedback/" in prompt
assert "evopolicygym submit program --episodes N" in prompt
assert "evopolicygym finish SUBMISSION_ID" in prompt
assert os.environ["CODEX_API_KEY"] == {api_key!r}
assert "EVOPOLICYGYM_TEST_SECRET" not in os.environ

program = workspace / "program"
assert (program / "policy.py").is_file()
(program / "policy.py").write_text({improved_source!r}, encoding="utf-8")


def call(*arguments):
    completed = subprocess.run(
        ["evopolicygym", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


submission = call("submit", "program", "--episodes", "1")
latest = json.loads((workspace / "feedback" / "latest.json").read_text())
feedback = json.loads(
    (workspace / "feedback" / latest["feedback"]).read_text()
)
assert feedback["score"] == 1.0
call("finish", submission["result"]["submission_id"])
print(json.dumps({{"type": "thread.started", "thread_id": "fake"}}))
print(json.dumps({{"type": "turn.completed"}}))
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            program = captured_program(root, initial_source)
            fake_codex = root / "fake-codex"
            fake_codex.write_text(fake_codex_source, encoding="utf-8")
            fake_codex.chmod(0o700)
            run_directory = root / "run-record"

            with patch.dict(
                "os.environ",
                {
                    "CODEX_API_KEY": api_key,
                    "EVOPOLICYGYM_TEST_SECRET": "must-not-be-inherited",
                },
            ):
                result = run(
                    program,
                    RecordingBenchmark(),
                    agent=Codex(
                        model="fake-model",
                        executable=str(fake_codex),
                    ),
                    execution=ProcessExecution.unsafe(),
                    record_to=run_directory,
                    config=RunConfig(
                        max_submissions=1,
                        episode_budget=1,
                        agent_timeout_seconds=10,
                    ),
                )

            invocation = json.loads(
                (run_directory / "agent" / "invocation.json").read_text()
            )
            manifest = json.loads((run_directory / "run.json").read_text())
            instructions = (
                run_directory / "agent" / "instructions.md"
            ).read_text()
            stderr = (run_directory / "agent" / "stderr.log").read_text()
            stdout_lines = tuple(
                json.loads(line)
                for line in (
                    run_directory / "agent" / "stdout.log"
                ).read_text().splitlines()
            )
            retained_documents = b"".join(
                path.read_bytes()
                for path in (
                    run_directory / "agent" / "invocation.json",
                    run_directory / "agent" / "instructions.md",
                    run_directory / "agent" / "stdout.log",
                    run_directory / "agent" / "stderr.log",
                    run_directory / "run.json",
                )
            )

        self.assertEqual(result.terminal_reason, "finished", stderr)
        self.assertEqual(result.final_submission_id, "submission-000001")
        self.assertIsNotNone(result.final_program)
        assert result.final_program is not None
        self.assertEqual(
            result.final_program.read_bytes("policy.py"),
            improved_source.encode(),
        )
        self.assertEqual(
            invocation["agent"],
            {"provider": "codex", "model": "fake-model"},
        )
        self.assertEqual(invocation["cwd"], "workspace")
        self.assertEqual(invocation["command"][-1], "@agent/instructions.md")
        self.assertEqual(invocation["instructions"], "agent/instructions.md")
        self.assertEqual(
            invocation["stdout_media_type"],
            "application/x-ndjson",
        )
        self.assertIn("CODEX_API_KEY", invocation["environment"]["inherited_allowlist"])
        self.assertNotIn(
            "EVOPOLICYGYM_TEST_SECRET",
            invocation["environment"]["inherited_allowlist"],
        )
        self.assertEqual(manifest["agent"]["provider"], "codex")
        self.assertEqual(manifest["agent"]["model"], "fake-model")
        self.assertEqual(
            manifest["workspace"]["program"],
            "workspace/program",
        )
        self.assertEqual(
            manifest["workspace"]["feedback"],
            "workspace/feedback",
        )
        self.assertIn("program/", instructions)
        self.assertEqual(
            [line["type"] for line in stdout_lines],
            ["thread.started", "turn.completed"],
        )
        self.assertNotIn(api_key.encode(), retained_documents)


class ConformanceTests(unittest.TestCase):
    def test_fixture_replay_checks_structural_determinism(self) -> None:
        benchmark = CounterBenchmark()
        fixture = BenchmarkFixture(
            episode=EpisodeSpec(environment_seed=5),
            actions=(1,),
        )

        report = check_benchmark(benchmark, fixtures=(fixture,))

        self.assertTrue(report.passed)
        report.raise_for_errors()


if __name__ == "__main__":
    unittest.main()
