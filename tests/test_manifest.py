from __future__ import annotations

import unittest
from dataclasses import dataclass

from evopolicygym.envs import manifest
from evopolicygym.envs.discover import Family, Report
from evopolicygym.envs.gym.spec import GymSpec


@dataclass(frozen=True, slots=True)
class BulkStub:
    spec: GymSpec
    family: str

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def upstream(self) -> str:
        return self.spec.id


class ManifestTest(unittest.TestCase):
    def test_static_manifest_tracks_builtins_and_p1_gym(self) -> None:
        entries = manifest.by_name()

        self.assertEqual(entries["toy"].level, manifest.Level.scored)
        self.assertEqual(entries["cartpole"].upstream, "CartPole-v1")
        self.assertEqual(entries["gym/taxi"].level, manifest.Level.tasked)
        self.assertEqual(entries["gym/taxi"].dependency, "env-gym")
        self.assertEqual(entries["gym/pendulum"].adapter, "gymnasium")
        self.assertEqual(entries["gym/lunar"].level, manifest.Level.smoke)
        self.assertEqual(entries["gym/lunar"].family, "Official Gymnasium: Box2D")
        self.assertEqual(entries["gym/racing"].level, manifest.Level.tasked)
        self.assertIn("external image", entries["gym/racing"].notes)
        self.assertEqual(entries["gym/halfcheetah5"].level, manifest.Level.smoke)
        self.assertEqual(entries["gym/halfcheetah5"].family, "Official Gymnasium: MuJoCo")

    def test_level_parse_accepts_codes_and_names(self) -> None:
        self.assertEqual(manifest.Level.parse("L2"), manifest.Level.tasked)
        self.assertEqual(manifest.Level.parse("scored"), manifest.Level.scored)
        self.assertEqual(manifest.Level.parse(1), manifest.Level.smoke)

    def test_discovery_entries_skip_known_upstreams(self) -> None:
        report = Report(
            packages=(),
            families=(
                Family(
                    name="Official Gymnasium: Classic Control",
                    count=2,
                    ids=("Pendulum-v1", "CartPole-v0"),
                    source="test",
                ),
            ),
            blocked=(),
        )

        rows = manifest.from_discovery(report)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "CartPole-v0")
        self.assertEqual(rows[0].level, manifest.Level.catalogued)
        self.assertEqual(rows[0].dependency, "env-gym")

    def test_bulk_specs_become_l1_manifest_entries(self) -> None:
        rows = manifest.from_bulk(
            (
                BulkStub(
                    GymSpec(name="gymnasium/CartPole-v1", id="CartPole-v1", steps=500),
                    "Official Gymnasium: Classic Control",
                ),
            )
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "gymnasium/CartPole-v1")
        self.assertEqual(rows[0].level, manifest.Level.smoke)
        self.assertEqual(rows[0].dependency, "env-gym")
        self.assertEqual(rows[0].upstream, "CartPole-v1")

    def test_bulk_jax_specs_use_jax_extra(self) -> None:
        rows = manifest.from_bulk(
            (
                BulkStub(
                    GymSpec(
                        name="gymnasium/phys2d/CartPole-v1",
                        id="phys2d/CartPole-v1",
                        steps=500,
                    ),
                    "Official Gymnasium: JAX",
                ),
            )
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].dependency, "env-jax")

    def test_mario_specs_use_mario_extra(self) -> None:
        rows = manifest.from_bulk(
            (
                BulkStub(
                    GymSpec(name="gymnasium/mo-supermario-v0", id="mo-supermario-v0", steps=500),
                    "MO-Gymnasium",
                ),
            )
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].dependency, "env-mario")


if __name__ == "__main__":
    unittest.main()
