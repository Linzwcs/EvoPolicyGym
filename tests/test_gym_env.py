from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evopolicygym import Case
from evopolicygym.check import check_env
from evopolicygym.envs import pendulum, registry, taxi
from evopolicygym.envs.gym.minigrid_assets import patch_minigrid_wfc_assets
from evopolicygym.envs.gym.world import _reward, _seed

HAS_GYM = importlib.util.find_spec("gymnasium") is not None
HAS_BROWSERGYM = importlib.util.find_spec("browsergym") is not None
HAS_MINIGRID = importlib.util.find_spec("minigrid") is not None

P1_NAMES = (
    "gym/acrobot",
    "gym/cartpole",
    "gym/mountaincar",
    "gym/continuouscar",
    "gym/pendulum",
    "gym/blackjack",
    "gym/cliff",
    "gym/frozenlake",
    "gym/taxi",
)


class MiniwobUrlTest(unittest.TestCase):
    def test_miniwob_env_url_gets_trailing_slash(self) -> None:
        from evopolicygym.envs.gym.dynamic import _miniwob_base_url

        with patch.dict(os.environ, {"MINIWOB_URL": "file:///tmp/miniwob"}, clear=False):
            self.assertEqual(_miniwob_base_url(), "file:///tmp/miniwob/")

    def test_miniwob_url_is_discovered_from_local_third_party_tree(self) -> None:
        from evopolicygym.envs.gym.dynamic import _miniwob_base_url

        with tempfile.TemporaryDirectory() as root:
            html = Path(root) / "third_party/miniwob-plusplus/miniwob/html/miniwob"
            html.mkdir(parents=True)
            (html / "click-button.html").write_text("", encoding="utf-8")
            (html / "ascending-numbers.html").write_text("", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(_miniwob_base_url(cwd=Path(root)), html.resolve().as_uri() + "/")


@unittest.skipUnless(HAS_GYM, "gymnasium extra is not installed")
class GymEnvTest(unittest.TestCase):
    def test_pendulum_env_resets_and_steps(self) -> None:
        env = pendulum()

        self.assertEqual(env.task.name, "gym/pendulum")
        self.assertEqual(env.task.steps, 200)
        self.assertEqual(env.task.obs["type"], "Box")
        self.assertEqual(env.task.act["type"], "Box")
        self.assertIn("# Pendulum", env.text)
        self.assertIn("cos(theta)", env.text)
        self.assertIn("torque", env.text)
        self.assertTrue(check_env(env).ok)

        case = Case(id=0, ref="gym/pendulum/train/000000", data={"seed": 123})
        world = env.make()
        obs = world.reset(case)
        same = env.make().reset(case)
        turn = world.step(world.sample())

        self.assertEqual(obs, same)
        self.assertEqual(len(obs), 3)
        self.assertEqual(len(turn.obs), 3)
        self.assertIsInstance(turn.reward, float)

    def test_taxi_env_resets_and_steps(self) -> None:
        env = taxi()

        self.assertEqual(env.task.name, "gym/taxi")
        self.assertEqual(env.task.obs["type"], "Discrete")
        self.assertEqual(env.task.act["type"], "Discrete")
        self.assertIn("# Taxi", env.text)
        self.assertIn("passenger", env.text)
        self.assertIn("pickup", env.text)
        self.assertTrue(check_env(env).ok)

        case = Case(id=0, ref="gym/taxi/train/000000", data={"seed": 456})
        world = env.make()
        obs = world.reset(case)
        turn = world.step(world.sample())

        self.assertIsInstance(obs, int)
        self.assertIsInstance(turn.obs, int)
        self.assertIsInstance(turn.reward, float)

    def test_default_registry_contains_gym_aliases(self) -> None:
        catalog = registry()

        for name in P1_NAMES:
            with self.subTest(name=name):
                self.assertEqual(catalog.get(name).task.name, name)

    def test_p1_envs_smoke(self) -> None:
        catalog = registry()

        for name in P1_NAMES:
            with self.subTest(name=name):
                env = catalog.get(name)
                report = check_env(env)
                self.assertTrue(report.ok, report.issues)
                case = Case(id=0, ref=f"{name}/train/000000", data={"seed": 101})
                world = env.make()
                obs = world.reset(case)
                same = env.make().reset(case)
                turn = world.step(world.sample())
                self.assertEqual(obs, same)
                self.assertIsInstance(turn.reward, float)

    def test_p1_envs_have_concrete_task_text(self) -> None:
        catalog = registry()

        expectations = {
            "gym/acrobot": "two-link",
            "gym/cartpole": "pole",
            "gym/mountaincar": "underpowered car",
            "gym/continuouscar": "continuous throttle",
            "gym/pendulum": "cos(theta)",
            "gym/blackjack": "usable ace",
            "gym/cliff": "cliff",
            "gym/frozenlake": "slippery",
            "gym/taxi": "dropoff",
        }
        for name, phrase in expectations.items():
            with self.subTest(name=name):
                text = catalog.get(name).text
                self.assertIn("## Objective", text)
                self.assertIn("## Observation", text)
                self.assertIn("## Action", text)
                self.assertIn("## Reward", text)
                self.assertIn(phrase, text)

    def test_native_dependency_envs_are_registered_when_available(self) -> None:
        catalog = registry()
        names = set(catalog.list())
        expectations = {
            "gym/lunar": ("LunarLander-v3", "Discrete"),
            "gym/reacher5": ("Reacher-v5", "Box"),
        }

        for name, (upstream, act_type) in expectations.items():
            if name not in names:
                continue
            with self.subTest(name=name):
                env = catalog.get(name)
                self.assertEqual(env.task.name, name)
                self.assertEqual(env.task.act["type"], act_type)
                self.assertIn(upstream, env.text)
                self.assertIn("## Objective", env.text)
                self.assertIn("## Observation", env.text)
                self.assertIn("## Action", env.text)

    def test_visual_gym_env_uses_external_observation_storage_when_available(self) -> None:
        catalog = registry()
        if "gym/racing" not in set(catalog.list()):
            return

        env = catalog.get("gym/racing")

        self.assertEqual(env.task.obs["type"], "Image")
        self.assertEqual(env.task.storage, "external")
        self.assertTrue(env.caps.observations)
        self.assertIn("Car Racing", env.text)
        self.assertIn("observations.npy", env.text)
        self.assertIn("steering", env.text)
        self.assertIn("## Reward", env.text)

    def test_vector_gym_reward_is_scalarized_for_evopolicygym_turns(self) -> None:
        scalar, raw = _reward([1.0, -0.25, 2])

        self.assertEqual(scalar, 2.75)
        self.assertEqual(raw, [1.0, -0.25, 2])

    def test_gym_seed_is_normalized_to_uint32_range(self) -> None:
        explicit = Case(id=0, ref="gym/test/train/000000", data={"seed": 2**40 + 5})
        generated = Case(id=1, ref="gym/test/train/000001")

        self.assertEqual(_seed(explicit), 5)
        self.assertGreaterEqual(_seed(generated), 0)
        self.assertLess(_seed(generated), 2**32)

    def test_bulk_registry_exposes_stable_long_names_only_when_requested(self) -> None:
        default = registry()

        self.assertNotIn("gymnasium/CartPole-v1", set(default.list()))

        bulk = registry(bulk=True, filters=("gymnasium/CartPole-v1",))
        env = bulk.get("gymnasium/CartPole-v1")

        self.assertEqual(env.task.name, "gymnasium/CartPole-v1")
        self.assertEqual(env.task.obs["type"], "Box")
        self.assertEqual(env.task.act["type"], "Discrete")
        self.assertIn("CartPole-v1", env.text)
        self.assertIn("classic-control", env.text)

    @unittest.skipUnless(HAS_BROWSERGYM, "browsergym extra is not installed")
    def test_browsergym_openended_bulk_spec_has_smoke_start_url(self) -> None:
        from evopolicygym.envs.gym.dynamic import specs

        (spec,) = specs(("browsergym/openended",))

        self.assertEqual(spec.name, "gymnasium/browsergym/openended")
        self.assertEqual(spec.kwargs["task_kwargs"]["start_url"], "about:blank")

    @unittest.skipUnless(HAS_MINIGRID, "minigrid extra is not installed")
    def test_minigrid_wfc_asset_patch_uses_vendored_patterns(self) -> None:
        from minigrid.envs.wfc import config

        preset = config.WFC_PRESETS_ALL["MazeSimple"]
        original = preset.pattern_path
        missing = Path(original).with_name("missing-evopolicygym-test.png")
        try:
            preset.pattern_path = missing

            patch_minigrid_wfc_assets("MiniGrid-WFC-MazeSimple-v0")

            patched = Path(preset.pattern_path)
            self.assertEqual(patched.name, "SimpleMaze.png")
            self.assertTrue(patched.exists())
            self.assertIn("evopolicygym", str(patched))
        finally:
            preset.pattern_path = original


if __name__ == "__main__":
    unittest.main()
