from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evopolicygym.envs.discover import Report, Spec, classify, write_json, write_markdown


class DiscoverTest(unittest.TestCase):
    def test_classify_groups_known_registry_ids(self) -> None:
        families = classify(
            (
                Spec("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv"),
                Spec("Taxi-v3", "gymnasium.envs.toy_text.taxi:TaxiEnv"),
                Spec("LunarLander-v3", "gymnasium.envs.box2d.lunar_lander:LunarLander"),
                Spec("HalfCheetah-v5", "gymnasium.envs.mujoco.half_cheetah_v5:HalfCheetahEnv"),
                Spec("phys2d/CartPole-v1", "gymnasium.envs.phys2d.cartpole:CartPoleJaxEnv"),
                Spec("ALE/Pong-v5", "ale_py.env:AtariEnv"),
                Spec("MiniGrid-DoorKey-5x5-v0", "minigrid.envs.doorkey:DoorKeyEnv"),
                Spec("merge-v0", "highway_env.envs.merge_env:MergeEnv"),
                Spec("FetchReach-v4", "gymnasium_robotics.envs.fetch.reach:FetchReachEnv"),
                Spec("deep-sea-treasure-v0", "mo_gymnasium.envs.deep_sea_treasure:DeepSeaTreasure"),
                Spec("browsergym/miniwob.click-button-v1", "browsergym.core.registration:make_env"),
            )
        )
        counts = {family.name: family.count for family in families}

        self.assertEqual(counts["Official Gymnasium: Classic Control"], 1)
        self.assertEqual(counts["Official Gymnasium: Toy Text"], 1)
        self.assertEqual(counts["Official Gymnasium: Box2D"], 1)
        self.assertEqual(counts["Official Gymnasium: MuJoCo"], 1)
        self.assertEqual(counts["Official Gymnasium: JAX"], 1)
        self.assertEqual(counts["Official Gymnasium: Atari / ALE"], 1)
        self.assertEqual(counts["MiniGrid"], 1)
        self.assertEqual(counts["HighwayEnv"], 1)
        self.assertEqual(counts["Gymnasium-Robotics"], 1)
        self.assertEqual(counts["MO-Gymnasium"], 1)
        self.assertEqual(counts["BrowserGym MiniWoB++"], 1)

    def test_writers_emit_files(self) -> None:
        families = classify((Spec("CartPole-v1", "gymnasium.envs.classic_control.cartpole:CartPoleEnv"),))
        report = Report(packages=(), families=families, blocked=())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            write_json(report, root / "discovered.json")
            write_markdown(report, root / "env_list.md")

            self.assertIn('"total": 1', (root / "discovered.json").read_text(encoding="utf-8"))
            self.assertIn(
                "- [x] CartPole-v1",
                (root / "env_list.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
