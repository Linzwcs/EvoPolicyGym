from __future__ import annotations

import unittest

from evopolicygym.authoring import (
    BenchmarkFixture,
    EpisodeSpec,
    InvalidAction,
    check_benchmark,
)
from evopolicygym.policy import PolicyValue, TensorValue

from jumanji_benchmarks import (
    JUMANJI_PROFILES,
    JumanjiBenchmark,
    JumanjiConfig,
    baseline_program,
)


class JumanjiBenchmarkTests(unittest.TestCase):
    def test_all_profiles_reset_and_take_one_strict_action(self) -> None:
        self.assertEqual(len(JUMANJI_PROFILES), 18)
        for profile in JUMANJI_PROFILES:
            with self.subTest(profile=profile):
                config = JumanjiConfig(profile=profile)
                environment = JumanjiBenchmark(config).make_environment(
                    EpisodeSpec(environment_seed=123)
                )
                try:
                    observation = environment.reset()
                    self.assertIsInstance(observation, dict)
                    step = environment.step(_first_valid_action(observation, config=config))
                    self.assertIsInstance(step.reward, float)
                finally:
                    environment.close()
                    environment.close()

    def test_profile_changes_public_identity(self) -> None:
        maze = JumanjiBenchmark()
        tetris = JumanjiBenchmark(JumanjiConfig(profile="tetris"))
        self.assertNotEqual(maze.spec.environment_digest, tetris.spec.environment_digest)
        self.assertEqual(tetris.spec.environment_parameters["profile"], "tetris")
        self.assertEqual(tetris.spec.max_episode_steps, 400)

    def test_invalid_profile_and_actions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JumanjiConfig(profile="unknown")
        with self.assertRaises(TypeError):
            JumanjiConfig(profile=1)  # type: ignore[arg-type]

        environment = JumanjiBenchmark().make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step(True)
        finally:
            environment.close()

        environment = JumanjiBenchmark(
            JumanjiConfig(profile="minesweeper")
        ).make_environment(EpisodeSpec(environment_seed=1))
        try:
            environment.reset()
            with self.assertRaises(InvalidAction):
                environment.step([10, 0])
        finally:
            environment.close()

    def test_masked_action_is_rejected_before_upstream_step(self) -> None:
        config = JumanjiConfig(profile="minesweeper")
        environment = JumanjiBenchmark(config).make_environment(
            EpisodeSpec(environment_seed=123)
        )
        try:
            observation = environment.reset()
            action = _first_valid_action(observation, config=config)
            step = environment.step(action)
            if not step.done:
                with self.assertRaises(InvalidAction):
                    environment.step(action)
        finally:
            environment.close()

    def test_episode_scenario_cannot_override_profile(self) -> None:
        with self.assertRaises(ValueError):
            JumanjiBenchmark().make_environment(
                EpisodeSpec(environment_seed=1, scenario={"profile": "tetris"})
            )

    def test_baseline_is_packaged(self) -> None:
        program = baseline_program()
        self.assertIn("policy.py", program.files)

    def test_replay_conformance(self) -> None:
        report = check_benchmark(
            JumanjiBenchmark(JumanjiConfig(profile="rubiks-cube-partly-scrambled")),
            fixtures=(
                BenchmarkFixture(
                    EpisodeSpec(environment_seed=123),
                    ([0, 0, 0],),
                ),
            ),
        )
        self.assertTrue(report.passed, report.issues)


def _first_valid_action(observation: PolicyValue, *, config: JumanjiConfig) -> PolicyValue:
    if type(observation) is not dict:
        raise AssertionError("expected an object observation")
    mask = observation.get("action_mask")
    if config.action_kind == "discrete":
        if type(mask) is not TensorValue or mask.dtype != "bool":
            if not config.has_action_mask:
                return 0
            raise AssertionError("expected a boolean action mask")
        return _first(mask.data)
    if not config.has_action_mask:
        return [0] * len(config.action_num_values)
    if type(mask) is not TensorValue or mask.dtype != "bool":
        raise AssertionError("expected a boolean action mask")
    if mask.shape == config.action_num_values:
        flat_index = _first(mask.data)
        return _unravel(flat_index, mask.shape)
    if (
        len(set(config.action_num_values)) == 1
        and mask.shape == (len(config.action_num_values), config.action_num_values[0])
    ):
        width = config.action_num_values[0]
        return [
            _first(mask.data[index * width : (index + 1) * width])
            for index in range(len(config.action_num_values))
        ]
    raise AssertionError(f"unexpected action mask shape: {mask.shape}")


def _first(values: bytes) -> int:
    for index, valid in enumerate(values):
        if valid:
            return index
    raise AssertionError("action mask has no valid action")


def _unravel(flat_index: int, shape: tuple[int, ...]) -> PolicyValue:
    result: list[PolicyValue] = [0 for _ in shape]
    for index in range(len(shape) - 1, -1, -1):
        flat_index, coordinate = divmod(flat_index, shape[index])
        result[index] = coordinate
    return result


if __name__ == "__main__":
    unittest.main()
