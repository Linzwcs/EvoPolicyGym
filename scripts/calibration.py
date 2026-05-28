"""Calibration sweep: run the reference PD agent at multiple episode
budgets on Pendulum and tabulate final_score + auxiliary metrics.

Usage::

    .venv/bin/python scripts/calibration.py
    .venv/bin/python scripts/calibration.py --budgets 8 32 128 --runs 2

Findings inform docs/findings.md (Day 14). Held-out is identical
across budgets (same env, same heldout.json), so the only thing
that should change is in-loop coverage and auxiliary metrics.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_POLICY = REPO_ROOT / "agents" / "pd_pendulum" / "policy.py"

# Make src/ importable when run directly from a source checkout.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from hlbench.core.server import Server  # noqa: E402


def _run_one(*, budget: int, max_per_submit: int, model: str) -> dict:
    """Run a single hlbench run at the given budget; return summary dict."""
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "run"
        srv = Server(
            env_id="pendulum",
            workspace_dir=ws,
            model=model,
            config_overrides={
                "episode_budget": budget,
                "max_episodes_per_submit": max_per_submit,
            },
        )
        shutil.copy(REFERENCE_POLICY, ws / "system" / "policy.py")

        # Submit in chunks of max_per_submit until budget is exhausted.
        submit_means: list[float] = []
        ei = 0
        while srv.info()["state"]["remaining_budget"] > 0:
            remaining = srv.info()["state"]["remaining_budget"]
            n_this = min(max_per_submit, remaining)
            ids = list(range(ei, ei + n_this))
            ei += n_this
            result = srv.submit(ids)
            if result.status == "ok":
                submit_means.append(result.summary["mean_return"])

        final = srv.finalize()
        return {
            "budget": budget,
            "max_per_submit": max_per_submit,
            "n_submits": len(submit_means),
            "in_loop_mean_of_means": (
                statistics.fmean(submit_means) if submit_means else None
            ),
            "in_loop_last_mean": submit_means[-1] if submit_means else None,
            "final_score": final.final_score,
            "held_out_mean": final.held_out_mean_return,
            "held_out_std": final.held_out_std_return,
            "auxiliary": json.loads(
                final.run_json_path.read_text()
            )["outcome"]["auxiliary"],
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--budgets", type=int, nargs="+",
        default=[8, 16, 32, 64, 128, 256],
        help="episode budgets to sweep",
    )
    p.add_argument(
        "--max-per-submit", type=int, default=8,
        help="cap on episodes per submit (kept constant across budgets so "
             "small-budget runs aren't penalized by the in-loop sample size)",
    )
    p.add_argument("--runs", type=int, default=1, help="repeats per budget (averaged)")
    p.add_argument("--model", default="reference-pd")
    p.add_argument("--out", default=None, help="optional path to write JSON results")
    args = p.parse_args()

    rows: list[dict] = []
    for budget in args.budgets:
        results = [
            _run_one(
                budget=budget,
                max_per_submit=min(args.max_per_submit, budget),
                model=args.model,
            )
            for _ in range(args.runs)
        ]
        # Average across repeats for the noisy metrics.
        scores = [r["final_score"] for r in results]
        held_outs = [r["held_out_mean"] for r in results]
        in_loops = [r["in_loop_mean_of_means"] for r in results
                    if r["in_loop_mean_of_means"] is not None]
        row = {
            "budget": budget,
            "n_submits": results[0]["n_submits"],
            "final_score": round(statistics.fmean(scores), 2),
            "final_score_std": round(
                statistics.stdev(scores) if len(scores) > 1 else 0.0, 2
            ),
            "held_out_mean": round(statistics.fmean(held_outs), 2),
            "in_loop_mean_of_means": (
                round(statistics.fmean(in_loops), 2) if in_loops else None
            ),
            "auc_in_loop": results[0]["auxiliary"]["auc_in_loop"],
            "episodes_to_50pct": results[0]["auxiliary"]["episodes_to_50pct"],
            "episodes_to_80pct": results[0]["auxiliary"]["episodes_to_80pct"],
            "held_out_gap": results[0]["auxiliary"]["held_out_gap"],
        }
        rows.append(row)
        print(
            f"budget={budget:>3}  n_sub={row['n_submits']:>2}  "
            f"final_score={row['final_score']:>5.1f}  "
            f"in_loop≈{row['in_loop_mean_of_means'] or 'n/a':>7}  "
            f"held_out={row['held_out_mean']:>7.1f}  "
            f"auc={row['auc_in_loop'] or 'n/a':>5}  "
            f"to50pct={row['episodes_to_50pct']}  "
            f"to80pct={row['episodes_to_80pct']}"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {len(rows)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
