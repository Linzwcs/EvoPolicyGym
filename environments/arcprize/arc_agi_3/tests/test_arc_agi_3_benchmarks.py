from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy
from arc_agi import Arcade, OperationMode  # type: ignore[import-untyped]
from arcengine import FrameDataRaw, GameAction, GameState
from evopolicygym import EvaluationConfig, evaluate
from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeRecord,
    EpisodeSpec,
    InvalidAction,
    Transition,
    check_benchmark,
)
from evopolicygym.execution import ProcessExecution
from evopolicygym.policy import PolicyValue, TensorValue
from PIL import Image

from arc_agi_3_benchmarks import (
    ARC_AGI_3_PUBLIC_GAMES,
    ArcAgi3Benchmark,
    ArcAgi3Config,
    baseline_program,
)
from arc_agi_3_benchmarks._upstream import EnvironmentWrapperLike


class _FakeWrapper:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.observation_space: FrameDataRaw | None = _response(
            game_id,
            actions=[1, 6],
        )
        self.reset_calls = 0
        self.steps: list[tuple[GameAction, dict[str, Any] | None]] = []

    def reset(self) -> FrameDataRaw:
        self.reset_calls += 1
        response = _response(
            self.game_id,
            state=GameState.NOT_FINISHED,
            actions=[1, 6],
        )
        self.observation_space = response
        return response

    def step(
        self,
        action: GameAction,
        data: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> FrameDataRaw:
        del reasoning
        self.steps.append((action, data))
        response = _response(
            self.game_id,
            state=(GameState.WIN if action is GameAction.ACTION6 else GameState.NOT_FINISHED),
            levels_completed=(2 if action is GameAction.ACTION6 else 1),
            win_levels=2,
            actions=[1, 6],
        )
        self.observation_space = response
        return response


class _FakeArcade:
    def __init__(self) -> None:
        self.created_scorecards = 0
        self.makes: list[tuple[str, int, str | None]] = []
        self.wrappers: list[_FakeWrapper] = []
        self.closed: list[str | None] = []

    def create_scorecard(
        self,
        source_url: str | None = None,
        tags: list[str] | None = None,
        opaque: Any | None = None,
    ) -> str:
        del source_url, tags, opaque
        self.created_scorecards += 1
        return f"scorecard-{self.created_scorecards}"

    def make(
        self,
        game_id: str,
        seed: int = 0,
        scorecard_id: str | None = None,
        save_recording: bool = False,
        include_frame_data: bool = True,
        render_mode: str | None = None,
        renderer: Any | None = None,
    ) -> EnvironmentWrapperLike:
        del save_recording, include_frame_data, render_mode, renderer
        self.makes.append((game_id, seed, scorecard_id))
        wrapper = _FakeWrapper(game_id)
        self.wrappers.append(wrapper)
        return wrapper

    def close_scorecard(self, scorecard_id: str | None = None) -> Any:
        self.closed.append(scorecard_id)
        return SimpleNamespace(
            score=37.5,
            total_environments=2,
            total_environments_completed=1,
            total_levels=7,
            total_levels_completed=3,
            total_actions=19,
        )


class ArcAgi3BenchmarkTests(unittest.TestCase):
    def test_public_profile_pins_all_twenty_five_games(self) -> None:
        self.assertEqual(len(ARC_AGI_3_PUBLIC_GAMES), 25)
        self.assertEqual(len(set(ARC_AGI_3_PUBLIC_GAMES)), 25)
        self.assertTrue(all("-" in game_id for game_id in ARC_AGI_3_PUBLIC_GAMES))
        config = ArcAgi3Config()
        self.assertEqual(config.game_ids, ARC_AGI_3_PUBLIC_GAMES)
        parameters = ArcAgi3Benchmark(config).spec.environment_parameters
        self.assertNotIn("game_ids", parameters)
        self.assertNotIn("operation_mode", parameters)
        self.assertEqual(parameters["episode_seed_supplied"], True)

    def test_episode_plan_is_deterministic_and_covers_collection(self) -> None:
        benchmark = ArcAgi3Benchmark(_arcade=_FakeArcade())
        first = tuple(benchmark.episodes("train", seed=7, count=25))
        repeated = tuple(benchmark.episodes("train", seed=7, count=25))
        self.assertEqual(first, repeated)
        game_ids = {cast(dict[str, str], episode.scenario)["game_id"] for episode in first}
        self.assertEqual(game_ids, set(ARC_AGI_3_PUBLIC_GAMES))

    def test_cached_initial_frame_and_strict_actions(self) -> None:
        arcade = _FakeArcade()
        config = ArcAgi3Config(
            profile="custom",
            custom_game_ids=("zz99-deadbeef",),
        )
        benchmark = ArcAgi3Benchmark(config, _arcade=arcade)
        episode = benchmark.episodes("train", seed=3, count=1)[0]
        environment = benchmark.make_environment(episode)
        observation = environment.reset()
        self.assertIsInstance(observation, dict)
        assert isinstance(observation, dict)
        frames = observation["frames"]
        self.assertIsInstance(frames, TensorValue)
        assert isinstance(frames, TensorValue)
        self.assertEqual(frames.dtype, "int8")
        self.assertEqual(frames.shape, (1, 64, 64))
        self.assertEqual(arcade.wrappers[0].reset_calls, 0)

        first = environment.step({"action": 1})
        self.assertEqual(first.reward, 1.0)
        reset = environment.step({"action": 0})
        self.assertTrue(cast(dict[str, object], reset.metrics)["reset"])
        self.assertEqual(arcade.wrappers[0].reset_calls, 1)
        won = environment.step({"action": 6, "x": 12, "y": 34})
        self.assertTrue(won.terminated)
        self.assertEqual(
            arcade.wrappers[0].steps[-1],
            (GameAction.ACTION6, {"x": 12, "y": 34}),
        )
        environment.close()
        environment.close()

    def test_environment_passes_public_conformance_replay(self) -> None:
        game_id = "zz99-deadbeef"
        benchmark = ArcAgi3Benchmark(
            ArcAgi3Config(
                profile="custom",
                custom_game_ids=(game_id,),
            ),
            _arcade=_FakeArcade(),
        )
        report = check_benchmark(
            benchmark,
            fixtures=(
                BenchmarkFixture(
                    episode=EpisodeSpec(
                        environment_seed=1,
                        scenario={"game_id": game_id},
                    ),
                    actions=(
                        {"action": 1},
                        {"action": 0},
                        {"action": 6, "x": 12, "y": 34},
                    ),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)

    def test_invalid_actions_are_not_repaired(self) -> None:
        arcade = _FakeArcade()
        benchmark = ArcAgi3Benchmark(
            ArcAgi3Config(
                profile="custom",
                custom_game_ids=("zz99-deadbeef",),
            ),
            _arcade=arcade,
        )
        environment = benchmark.make_environment(
            EpisodeSpec(
                environment_seed=1,
                scenario={"game_id": "zz99-deadbeef"},
            )
        )
        try:
            environment.reset()
            invalid = (
                {"action": 2},
                {"action": 6},
                {"action": 6, "x": 64, "y": 0},
                {"action": 1, "x": 0},
                1,
            )
            for action in invalid:
                with self.subTest(action=action), self.assertRaises(InvalidAction):
                    environment.step(action)  # type: ignore[arg-type]
            self.assertEqual(arcade.wrappers[0].steps, [])
        finally:
            environment.close()

    def test_horizon_truncates_and_feedback_closes_shared_scorecard(self) -> None:
        arcade = _FakeArcade()
        config = ArcAgi3Config(
            profile="custom",
            custom_game_ids=("zz99-deadbeef",),
            max_episode_steps=1,
        )
        benchmark = ArcAgi3Benchmark(config, _arcade=arcade)
        episode = benchmark.episodes("validation", seed=8, count=1)[0]
        environment = benchmark.make_environment(episode)
        initial = environment.reset()
        step = environment.step({"action": 1})
        self.assertTrue(step.truncated)
        environment.close()

        feedback = benchmark.feedback(
            (
                EpisodeRecord(
                    episode=episode,
                    policy_seed=4,
                    initial_observation=initial,
                    transitions=(Transition(action={"action": 1}, step=step),),
                ),
            )
        )
        self.assertEqual(feedback.score, 37.5)
        content = cast(dict[str, object], feedback.content)
        self.assertEqual(content["total_actions"], 19)
        summaries = cast(list[dict[str, object]], content["episode_summaries"])
        self.assertEqual(summaries[0]["steps"], 1)
        self.assertEqual(summaries[0]["action_counts"], {"1": 1})
        final = cast(dict[str, object], summaries[0]["final_observation"])
        self.assertEqual(final["state"], "NOT_FINISHED")
        self.assertEqual(final["levels_completed"], 1)
        self.assertEqual(len(feedback.artifacts), 3)
        self.assertEqual(feedback.artifacts[0].name, "trace.jsonl")
        documents = [json.loads(line) for line in feedback.artifacts[0].read_bytes().splitlines()]
        self.assertEqual(
            [document["type"] for document in documents],
            ["episode", "observation", "transition", "observation"],
        )
        initial_frames = documents[1]["observation"]["frames"]
        result_frames = documents[3]["observation"]["frames"]
        self.assertEqual(initial_frames["dtype"], "int8")
        self.assertEqual(initial_frames["shape"], [1, 64, 64])
        self.assertEqual(initial_frames["encoding"], "numpy-npz")
        self.assertEqual(
            initial_frames["artifact"],
            "episode-000/observations.npz",
        )
        self.assertEqual(initial_frames["key"], "observation_000000")
        self.assertEqual(result_frames["key"], "observation_000001")
        self.assertEqual(len(initial_frames["sha256"]), 64)
        self.assertNotIn("last_frame_rows", documents[1]["observation"])
        self.assertEqual(documents[2]["decision_observation_index"], 0)
        self.assertEqual(documents[2]["result_observation_index"], 1)
        self.assertNotIn("decision_observation", documents[2])
        self.assertNotIn("result_observation", documents[2])

        videos = cast(list[dict[str, object]], content["videos"])
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["artifact"], "episode-000/playback.gif")
        self.assertEqual(videos[0]["source_animation_frames"], 2)
        self.assertEqual(videos[0]["encoded_frames"], 2)
        timeline = cast(list[dict[str, object]], videos[0]["timeline"])
        self.assertEqual(timeline[0]["decision_for_step_index"], 0)
        self.assertIsNone(timeline[0]["after_step_index"])
        self.assertEqual(timeline[1]["after_step_index"], 0)
        self.assertIsNone(timeline[1]["decision_for_step_index"])
        self.assertEqual(timeline[1]["state"], "NOT_FINISHED")

        observations = feedback.artifacts[1]
        self.assertEqual(
            observations.name,
            "episode-000/observations.npz",
        )
        self.assertEqual(observations.media_type, "application/x-npz")
        with zipfile.ZipFile(BytesIO(observations.read_bytes())) as archive:
            self.assertTrue(
                all(
                    member.date_time == (1980, 1, 1, 0, 0, 0)
                    for member in archive.infolist()
                )
            )
        with numpy.load(BytesIO(observations.read_bytes()), allow_pickle=False) as arrays:
            self.assertEqual(set(arrays.files), {"observation_000000", "observation_000001"})
            numpy.testing.assert_array_equal(
                arrays["observation_000000"],
                numpy.zeros((1, 64, 64), dtype=numpy.int8),
            )
            numpy.testing.assert_array_equal(
                arrays["observation_000001"],
                numpy.ones((1, 64, 64), dtype=numpy.int8),
            )

        video = feedback.artifacts[2]
        self.assertEqual(video.name, "episode-000/playback.gif")
        self.assertEqual(video.media_type, "image/gif")
        with Image.open(BytesIO(video.read_bytes())) as playback:
            self.assertEqual(playback.size, (256, 288))
            self.assertEqual(cast(Any, playback).n_frames, 2)
            self.assertEqual(playback.convert("RGB").getpixel((0, 0)), (255, 255, 255))
            playback.seek(1)
            self.assertEqual(playback.convert("RGB").getpixel((0, 0)), (204, 204, 204))

        public_bytes = json.dumps(content, sort_keys=True).encode() + b"".join(
            artifact.read_bytes() for artifact in feedback.artifacts
        )
        for private_text in (
            b"zz99-deadbeef",
            b"environment_seed",
            b"policy_seed",
            b"scenario",
            b"scorecard-1",
        ):
            self.assertNotIn(private_text, public_bytes)
        self.assertEqual(arcade.closed, ["scorecard-1"])

    def test_feedback_retains_every_animation_frame(self) -> None:
        arcade = _FakeArcade()
        benchmark = ArcAgi3Benchmark(
            ArcAgi3Config(
                profile="custom",
                custom_game_ids=("zz99-deadbeef",),
            ),
            _arcade=arcade,
        )
        episode = benchmark.episodes("train", seed=3, count=1)[0]
        environment = benchmark.make_environment(episode)
        environment.close()

        expected = numpy.stack(
            (
                numpy.zeros((64, 64), dtype=numpy.int8),
                numpy.full((64, 64), 7, dtype=numpy.int8),
                numpy.full((64, 64), 15, dtype=numpy.int8),
            )
        )
        observation: PolicyValue = {
            "frames": TensorValue(
                dtype="int8",
                shape=expected.shape,
                data=expected.tobytes(order="C"),
            ),
            "state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 2,
            "available_actions": [1, 6],
        }
        feedback = benchmark.feedback(
            (
                EpisodeRecord(
                    episode=episode,
                    policy_seed=9,
                    initial_observation=observation,
                    transitions=(),
                ),
            )
        )

        trace = [
            json.loads(line)
            for line in feedback.artifacts[0].read_bytes().splitlines()
        ]
        descriptor = trace[1]["observation"]["frames"]
        self.assertEqual(descriptor["shape"], [3, 64, 64])
        self.assertNotIn("last_frame_rows", trace[1]["observation"])
        with numpy.load(
            BytesIO(feedback.artifacts[1].read_bytes()),
            allow_pickle=False,
        ) as arrays:
            numpy.testing.assert_array_equal(
                arrays["observation_000000"],
                expected,
            )

    def test_same_game_supports_multiple_seeded_episodes(self) -> None:
        arcade = _FakeArcade()
        config = ArcAgi3Config(
            profile="custom",
            custom_game_ids=("aa00-00000000",),
        )
        benchmark = ArcAgi3Benchmark(config, _arcade=arcade)
        episodes = tuple(benchmark.episodes("test", seed=0, count=3))
        self.assertEqual(len({episode.environment_seed for episode in episodes}), 3)
        for episode in episodes:
            environment = benchmark.make_environment(episode)
            environment.reset()
            environment.close()
        self.assertEqual(
            [seed for _, seed, _ in arcade.makes],
            [episode.environment_seed for episode in episodes],
        )
        self.assertEqual(
            {scorecard_id for _, _, scorecard_id in arcade.makes},
            {"scorecard-1"},
        )
        records = tuple(
            EpisodeRecord(
                episode=episode,
                policy_seed=index,
                initial_observation=None,
                transitions=(),
                policy_failure="invalid_action",
            )
            for index, episode in enumerate(episodes)
        )
        benchmark.feedback(records)
        self.assertEqual(arcade.closed, ["scorecard-1"])
        repeated = benchmark.episodes("test", seed=1, count=1)[0]
        repeated_environment = benchmark.make_environment(repeated)
        repeated_environment.close()
        self.assertEqual(arcade.makes[-1][2], "scorecard-2")

    def test_feedback_saves_playback_for_every_episode(self) -> None:
        arcade = _FakeArcade()
        benchmark = ArcAgi3Benchmark(
            ArcAgi3Config(
                profile="custom",
                custom_game_ids=("zz99-deadbeef",),
                max_episode_steps=1,
            ),
            _arcade=arcade,
        )
        records: list[EpisodeRecord] = []
        for episode in benchmark.episodes("train", seed=18, count=5):
            environment = benchmark.make_environment(episode)
            try:
                initial = environment.reset()
                action: PolicyValue = {"action": 1}
                step = environment.step(action)
            finally:
                environment.close()
            records.append(
                EpisodeRecord(
                    episode=episode,
                    policy_seed=4,
                    initial_observation=initial,
                    transitions=(Transition(action=action, step=step),),
                )
            )

        feedback = benchmark.feedback(tuple(records))

        self.assertEqual(
            [
                artifact.name
                for artifact in feedback.artifacts
                if artifact.media_type == "image/gif"
            ],
            [f"episode-{index:03d}/playback.gif" for index in range(5)],
        )
        content = cast(dict[str, object], feedback.content)
        self.assertEqual(content["video_episodes"], 5)
        self.assertEqual(content["video_episode_results"], 5)
        self.assertEqual(content["video_episodes_without_gif"], 0)
        videos = cast(list[dict[str, object]], content["videos"])
        self.assertEqual(
            [video["status"] for video in videos],
            ["available"] * 5,
        )

    def test_invalid_configurations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ArcAgi3Config(profile="custom")
        with self.assertRaises(ValueError):
            ArcAgi3Config(max_episode_steps=0)
        with self.assertRaises(ValueError):
            ArcAgi3Config(
                profile="custom",
                custom_game_ids=("aa00-one", "aa00-two"),
            )
        with self.assertRaises(TypeError):
            ArcAgi3Config(custom_game_ids=["aa00-one"])  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ArcAgi3Config(operation_mode="online")  # type: ignore[call-arg]

    def test_baseline_is_packaged(self) -> None:
        self.assertIn("policy.py", baseline_program().files)

    def test_baseline_runs_through_public_evaluation(self) -> None:
        benchmark = ArcAgi3Benchmark(
            ArcAgi3Config(
                profile="custom",
                custom_game_ids=("zz99-deadbeef",),
                max_episode_steps=2,
            ),
            _arcade=_FakeArcade(),
        )
        result = evaluate(
            baseline_program(),
            benchmark,
            execution=ProcessExecution.unsafe(),
            config=EvaluationConfig(
                split="validation",
                episodes=1,
                seed=5,
                episode_timeout_seconds=10,
            ),
        )
        self.assertEqual(result.feedback.score, 37.5)
        self.assertEqual(result.episodes[0].steps, 2)

    def test_real_offline_toolkit_reset_and_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game_dir = root / "zz99" / "deadbeef"
            game_dir.mkdir(parents=True)
            metadata = {
                "game_id": "zz99-deadbeef",
                "title": "ZZ99",
                "class_name": "Zz99",
                "baseline_actions": [1],
            }
            (game_dir / "metadata.json").write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )
            (game_dir / "zz99.py").write_text(
                _LOCAL_GAME_SOURCE,
                encoding="utf-8",
            )
            arcade = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(root),
                recordings_dir=str(root / "recordings"),
            )
            benchmark = ArcAgi3Benchmark(
                ArcAgi3Config(
                    profile="custom",
                    custom_game_ids=("zz99-deadbeef",),
                    max_episode_steps=2,
                ),
                _arcade=arcade,
            )
            episode = benchmark.episodes("train", seed=9, count=1)[0]
            environment = benchmark.make_environment(episode)
            try:
                observation = environment.reset()
                self.assertIsInstance(observation, dict)
                step = environment.step({"action": 1})
                self.assertTrue(step.terminated)
                self.assertEqual(step.reward, 1.0)
            finally:
                environment.close()


def _response(
    game_id: str,
    *,
    state: GameState = GameState.NOT_FINISHED,
    levels_completed: int = 0,
    win_levels: int = 2,
    actions: list[int],
) -> FrameDataRaw:
    response = FrameDataRaw(
        game_id=game_id,
        state=state,
        levels_completed=levels_completed,
        win_levels=win_levels,
        available_actions=actions,
    )
    response.frame = [
        numpy.full((64, 64), levels_completed % 16, dtype=numpy.int8)
    ]
    return response


_LOCAL_GAME_SOURCE = """
from arcengine import ARCBaseGame, GameAction, Level, Sprite


class Zz99(ARCBaseGame):
    def __init__(self, seed=0):
        super().__init__(
            game_id="zz99-deadbeef",
            levels=[Level([Sprite([[1]], name="player")])],
            available_actions=[1],
            seed=seed,
        )

    def step(self):
        if self.action.id is GameAction.ACTION1:
            self.next_level()
        self.complete_action()
"""


if __name__ == "__main__":
    unittest.main()
