"""Generate train.json + heldout.json for an env.

Usage::

    python scripts/gen_seeds.py <env_dir>/data \\
        --master-seed 42 \\
        --n-train 256 --n-heldout 256

Per-env convention: seed pools live at
``src/hlbench/envs/<env_id>/data/{train,heldout}.json`` so the env
package can self-contain its training data (the env's ``__init__.py``
loads them via ``_HERE / "data" / "train.json"``).

Deterministic given the master seed. Held-out seeds are drawn from a
disjoint range relative to train (offset by 2**24) for transparency,
but the values themselves are sampled — agents can never see them, so
the disjointness is belt-and-suspenders.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def gen(master_seed: int, n: int, offset: int = 0) -> list[int]:
    rng = random.Random(master_seed + offset)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "env_dir", type=Path,
        help="Directory to write train.json/heldout.json into "
             "(typically src/hlbench/envs/<env_id>/data/)",
    )
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=256)
    p.add_argument("--n-heldout", type=int, default=256)
    args = p.parse_args()

    args.env_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = gen(args.master_seed, args.n_train, offset=0)
    heldout_seeds = gen(args.master_seed, args.n_heldout, offset=2**24)

    # Sanity: no overlap.
    assert set(train_seeds).isdisjoint(set(heldout_seeds)), "train/heldout overlap!"

    (args.env_dir / "train.json").write_text(
        json.dumps({"real_seeds": train_seeds}, indent=None) + "\n"
    )
    (args.env_dir / "heldout.json").write_text(
        json.dumps({"real_seeds": heldout_seeds}, indent=None) + "\n"
    )
    print(f"wrote {args.env_dir}/train.json   ({len(train_seeds)} seeds)")
    print(f"wrote {args.env_dir}/heldout.json ({len(heldout_seeds)} seeds)")


if __name__ == "__main__":
    main()
