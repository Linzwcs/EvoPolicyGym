from __future__ import annotations

import importlib.util
import unittest

from evopolicygym import Pool, PoolKind
from evopolicygym.envs import CartPole, Registry, cartpole, registry, toy


class EnvTest(unittest.TestCase):
    def test_pool_maps_ids_to_case_refs(self) -> None:
        pool = Pool(kind=PoolKind.train, size=8, ref="toy/train")

        case = pool.case(2)

        self.assertEqual(case.id, 2)
        self.assertEqual(case.ref, "toy/train/000002")
        self.assertTrue(pool.contains(7))
        self.assertFalse(pool.contains(8))
        with self.assertRaisesRegex(ValueError, "outside pool"):
            pool.case(8)

    def test_toy_env_exposes_task_and_world_factory(self) -> None:
        env = toy()

        self.assertEqual(env.task.name, "toy")
        self.assertEqual(env.task.cases, 8)
        self.assertIn("# Toy", env.text)
        self.assertIn("one-step additive", env.text)
        self.assertEqual(env.secret.valid, "toy/validation")
        self.assertEqual(env.secret.valid_size, 64)
        self.assertEqual(env.secret.final_size, 256)
        self.assertEqual(env.pool(PoolKind.train).size, 8)
        self.assertEqual(env.pool(PoolKind.valid).ref, "toy/validation")
        self.assertEqual(env.pool(PoolKind.final).size, 256)
        self.assertEqual(env.value(env.pool(PoolKind.final), (1.0, 3.0)), 2.0)

        world = env.make()
        obs = world.reset(Pool(kind=PoolKind.train, size=8, ref=env.secret.train).case(2))
        turn = world.step(1)

        self.assertEqual(obs, 2)
        self.assertEqual(turn.obs, 3)
        self.assertEqual(turn.reward, 3.0)
        self.assertTrue(turn.done)

    def test_cartpole_env_exposes_task_and_world_factory(self) -> None:
        env = cartpole()

        self.assertEqual(env.task.name, "cartpole")
        self.assertEqual(env.task.steps, 500)
        self.assertIn("# CartPole", env.text)
        self.assertIn("## Observation", env.text)
        self.assertEqual(env.task.cases, 64)
        self.assertEqual(env.task.obs["shape"], [4])
        self.assertEqual(env.task.act["n"], 2)
        self.assertEqual(env.secret.valid_size, 32)
        self.assertEqual(env.secret.final_size, 64)
        self.assertEqual(env.value(env.pool(PoolKind.train), (100.0,)), None)
        self.assertEqual(env.value(env.pool(PoolKind.valid), (500.0, 250.0)), 75.0)
        self.assertEqual(env.value(env.pool(PoolKind.final), (500.0, 250.0)), 75.0)

        case = env.pool(PoolKind.train).case(3)
        world = env.make()
        obs = world.reset(case)
        same = env.make().reset(case)
        turn = world.step(1)

        self.assertEqual(obs, same)
        self.assertEqual(len(obs), 4)
        self.assertEqual(len(turn.obs), 4)
        self.assertEqual(turn.reward, 1.0)
        self.assertFalse(turn.done)
        self.assertIn(world.sample(), (0, 1))

    def test_cartpole_truncates_at_max_steps(self) -> None:
        world = CartPole(max_steps=1)

        world.reset(Pool(kind=PoolKind.train, size=1, ref="cartpole/train").case(0))
        turn = world.step(0)

        self.assertTrue(turn.done)
        self.assertTrue(turn.truncated)

    def test_registry_lists_and_gets_envs(self) -> None:
        env = toy()
        catalog = Registry.of(env)

        self.assertEqual(catalog.list(), ("toy",))
        self.assertEqual(catalog.get("toy"), env)
        with self.assertRaisesRegex(KeyError, "unknown environment"):
            catalog.get("missing")

    def test_default_registry_contains_toy(self) -> None:
        catalog = registry()

        names = catalog.list()
        self.assertIn("cartpole", names)
        self.assertIn("toy", names)
        if importlib.util.find_spec("gymnasium") is not None:
            self.assertIn("gym/pendulum", names)
            self.assertIn("gym/taxi", names)


if __name__ == "__main__":
    unittest.main()
