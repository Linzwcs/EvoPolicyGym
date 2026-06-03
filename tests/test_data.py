from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evopolicygym import PoolKind
from evopolicygym.data import load, make
from evopolicygym.envs import toy


class DataTest(unittest.TestCase):
    def test_loads_external_case_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train", [{"start": 2}, {"start": 4}])
            _write(root, "valid", [{"start": 6}])
            _write(root, "heldout", [{"start": 8, "ref": "heldout/custom"}])

            corpus = load(root, env="toy")

            self.assertEqual(corpus.pool(PoolKind.train).size, 2)
            self.assertEqual(corpus.train.pool.case(1).data["start"], 4)
            self.assertEqual(corpus.final.pool.case(0).ref, "heldout/custom")
            self.assertEqual(corpus.versions()["data_train_hash"], _sha(root / "train.json"))

    def test_loads_array_splits_and_wraps_scalar_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.json").write_text(json.dumps([1, 2]), encoding="utf-8")
            (root / "valid.json").write_text(json.dumps([3]), encoding="utf-8")
            (root / "heldout.json").write_text(json.dumps([4]), encoding="utf-8")

            corpus = load(root)

            self.assertEqual(corpus.train.pool.case(0).data, {"value": 1})

    def test_rejects_overlap_and_bad_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, "train", [{"seed": 1}])
            _write(root, "valid", [{"seed": 1}])
            _write(root, "heldout", [{"seed": 2}])

            with self.assertRaisesRegex(ValueError, "overlaps"):
                load(root, env="toy")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.json").write_text(json.dumps({"bad": True}), encoding="utf-8")
            (root / "valid.json").write_text(json.dumps([]), encoding="utf-8")
            (root / "heldout.json").write_text(json.dumps([]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cases array"):
                load(root)

    def test_make_writes_configurable_seed_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"

            corpus = make(
                root,
                toy(),
                seed=7,
                train_size=3,
                valid_size=2,
                heldout_size=1,
            )

            self.assertEqual(corpus.train.pool.size, 3)
            self.assertEqual(corpus.valid.pool.size, 2)
            self.assertEqual(corpus.final.pool.size, 1)
            self.assertEqual(corpus.train.pool.ref, "toy/train")
            self.assertEqual(corpus.valid.pool.ref, "toy/validation")
            self.assertEqual(corpus.final.pool.ref, "toy/heldout")
            self.assertIn("seed", corpus.train.pool.case(0).data)
            self.assertEqual(len(_all_seeds(corpus)), 6)

            payload = json.loads((root / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["env"], "toy")
            self.assertEqual(payload["split"], "train")
            self.assertEqual(payload["generator"], {"kind": "seed", "seed": 7, "size": 3})

            with self.assertRaises(FileExistsError):
                make(root, toy(), seed=7)

    def test_make_is_reproducible_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = make(root / "a", toy(), seed=11, train_size=2, valid_size=1, heldout_size=1)
            second = make(root / "b", toy(), seed=11, train_size=2, valid_size=1, heldout_size=1)

            self.assertEqual(_all_seeds(first), _all_seeds(second))
            self.assertEqual(
                (root / "a" / "train.json").read_text(encoding="utf-8"),
                (root / "b" / "train.json").read_text(encoding="utf-8"),
            )


def _write(root: Path, split: str, cases: list[dict]) -> None:
    payload = {"env": "toy", "split": split, "cases": cases}
    (root / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _all_seeds(corpus) -> set[int]:
    return {
        case.data["seed"]
        for split in (corpus.train, corpus.valid, corpus.final)
        for case in split.pool.cases
    }


if __name__ == "__main__":
    unittest.main()
