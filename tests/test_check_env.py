from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from evopolicygym import Caps, Case, Env, Pool, PoolKind, Secret, Task, Turn
from evopolicygym.check import check_env
from evopolicygym.envs import cartpole, toy


class EnvCheckTest(unittest.TestCase):
    def test_check_env_accepts_toy_contract(self) -> None:
        report = check_env(toy())

        self.assertTrue(report.ok, report.issues)

    def test_check_env_accepts_cartpole_contract(self) -> None:
        report = check_env(cartpole())

        self.assertTrue(report.ok, report.issues)

    def test_check_env_reports_hidden_leak_and_missing_caps(self) -> None:
        env = Env(
            task=Task(
                name="leaky",
                version="0.1",
                obs={},
                act={},
                steps=1,
                cases=1,
                storage="external",
            ),
            secret=Secret(
                train="leaky/train",
                valid="leaky/validation",
                final="leaky/heldout",
                expert=1.0,
                random=0.0,
            ),
            make=GoodWorld,
            value=_value,
            text="hidden pool is leaky/validation",
        )

        report = check_env(env)

        self.assertIn("hidden_leak", _codes(report))
        self.assertIn("caps_observations", _codes(report))

    def test_check_env_reports_bad_world_boundary(self) -> None:
        env = Env(
            task=Task(name="bad", version="0.1", obs={}, act={}, steps=1, cases=1),
            secret=Secret(
                train="bad/train",
                valid="bad/validation",
                final="bad/heldout",
                expert=1.0,
                random=0.0,
            ),
            make=BadWorld,
            value=_value,
            text=_task_text(),
        )

        report = check_env(env)

        self.assertIn("world_reset", _codes(report))

    def test_check_env_reports_missing_final_value(self) -> None:
        env = Env(
            task=Task(name="novalue", version="0.1", obs={}, act={}, steps=1, cases=1),
            secret=Secret(
                train="novalue/train",
                valid="novalue/validation",
                final="novalue/heldout",
                expert=1.0,
                random=0.0,
            ),
            make=GoodWorld,
            value=lambda pool, returns: None,
            text=_task_text(),
        )

        report = check_env(env)

        self.assertIn("final_value", _codes(report))

    def test_check_env_reports_value_exception_without_raising(self) -> None:
        env = Env(
            task=Task(name="badvalue", version="0.1", obs={}, act={}, steps=1, cases=1),
            secret=Secret(
                train="badvalue/train",
                valid="badvalue/validation",
                final="badvalue/heldout",
                expert=1.0,
                random=0.0,
            ),
            make=GoodWorld,
            value=lambda pool, returns: (_ for _ in ()).throw(RuntimeError("bad value")),
            text=_task_text(),
        )

        report = check_env(env)

        self.assertIn("value", _codes(report))

    def test_check_env_accepts_external_storage_with_caps(self) -> None:
        env = Env(
            task=Task(
                name="pixels",
                version="0.1",
                obs={},
                act={},
                steps=1,
                cases=1,
                storage="external",
            ),
            secret=Secret(
                train="pixels/train",
                valid="pixels/validation",
                final="pixels/heldout",
                expert=1.0,
                random=0.0,
            ),
            make=GoodWorld,
            value=_value,
            caps=Caps(observations=True),
            text=_task_text(),
        )

        report = check_env(env)

        self.assertNotIn("caps_observations", _codes(report))

    def test_check_env_accepts_file_backed_case_sources(self) -> None:
        env = _source_env()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sources(root, train=[1, 2], valid=[3], final=[4])

            report = check_env(env, root)

            self.assertTrue(report.ok, report.issues)

    def test_check_env_reports_source_size_and_overlap(self) -> None:
        env = _source_env()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sources(root, train=[1], valid=[2], final=[4])

            report = check_env(env, root)

            self.assertIn("source_size", _codes(report))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sources(root, train=[1, 2], valid=[1], final=[4])

            report = check_env(env, root)

            self.assertIn("source_overlap", _codes(report))

    def test_check_env_reports_missing_and_bad_source_shape(self) -> None:
        env = _source_env()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "train.json", [1, 2])
            _write(root / "valid.json", [3])

            report = check_env(env, root)

            self.assertIn("source_missing", _codes(report))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "train.json", {"bad": True})
            _write(root / "valid.json", [3])
            _write(root / "heldout.json", [4])

            report = check_env(env, root)

            self.assertIn("source_shape", _codes(report))

    def test_check_env_reports_missing_task_text(self) -> None:
        env = Env(
            task=Task(name="notext", version="0.1", obs={}, act={}, steps=1, cases=1),
            secret=Secret(
                train="notext/train",
                valid="notext/validation",
                final="notext/heldout",
                expert=1.0,
                random=0.0,
            ),
            make=GoodWorld,
            value=_value,
        )

        report = check_env(env)

        self.assertIn("task_text", _codes(report))


@dataclass(slots=True)
class GoodWorld:
    state: int = 0

    def reset(self, case: Case) -> int:
        self.state = case.id
        return self.state

    def step(self, action: int) -> Turn:
        self.state += int(action)
        return Turn(obs=self.state, reward=float(self.state), terminated=True)

    def sample(self) -> int:
        return 0


class BadWorld(GoodWorld):
    def reset(self, case: Case) -> int:
        raise RuntimeError("bad reset")


def _value(pool: Pool, returns: tuple[float, ...]) -> float | None:
    if pool.kind == PoolKind.final:
        return sum(returns) / len(returns)
    return None


def _source_env() -> Env:
    return Env(
        task=Task(name="source", version="0.1", obs={}, act={}, steps=1, cases=2),
        secret=Secret(
            train="source/train",
            valid="source/validation",
            final="source/heldout",
            expert=1.0,
            random=0.0,
            valid_size=1,
            final_size=1,
        ),
        make=GoodWorld,
        value=_value,
        text=_task_text(),
    )


def _task_text() -> str:
    return "# Test\n\n## Objective\n\nDo the task.\n"


def _write_sources(root: Path, *, train, valid, final) -> None:
    _write(root / "train.json", train)
    _write(root / "valid.json", valid)
    _write(root / "heldout.json", final)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


if __name__ == "__main__":
    unittest.main()
