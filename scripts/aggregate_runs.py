"""Aggregate completed run.json files into a summary table.

Scans ``runs/<model-slug>/<env>/<exp-id>/run.json`` (the canonical
output layout from ``hlbench init / serve / finalize``) and produces
either a flat per-run table or a per-experiment aggregate.

Usage:

    # All completed runs across all (model, env, exp-id) triples
    .venv/bin/python scripts/aggregate_runs.py

    # Filter by exp-id (substring match)
    .venv/bin/python scripts/aggregate_runs.py --exp-id v1paper-sonnet-b256

    # Per-exp-id × per-env table (paper-friendly)
    .venv/bin/python scripts/aggregate_runs.py --exp-id v1paper-sonnet --pivot env

    # Compare exp-ids (e.g., budget=64 vs budget=256)
    .venv/bin/python scripts/aggregate_runs.py --pivot exp_id --env pendulum half_cheetah

    # CSV output
    .venv/bin/python scripts/aggregate_runs.py --format csv > runs_summary.csv

    # Errored / unfinished runs (no run.json or status != completed)
    .venv/bin/python scripts/aggregate_runs.py --include-errors

Output formats:
    --format table     (default; aligned text table for terminals)
    --format markdown  (for pasting into paper docs)
    --format csv       (for spreadsheet import)
    --format json      (for scripts downstream)

Pivot modes:
    --pivot env        rows=exp_id, cols=env, values=final_score
    --pivot exp_id     rows=env, cols=exp_id, values=final_score
    (default)          one row per (model, exp_id, env)
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def scan_runs(runs_root: Path) -> list[dict[str, Any]]:
    """Walk runs/<model>/<env>/<exp-id>/run.json and parse each.

    Returns one dict per run.json file with flattened fields. Runs with
    no run.json (unfinished/aborted) are NOT included; use --include-errors
    via the caller to collect those separately.
    """
    rows: list[dict[str, Any]] = []
    for run_json in runs_root.rglob("run.json"):
        # Expected path: runs/<model>/<env>/<exp-id>/run.json
        try:
            rel = run_json.relative_to(runs_root)
            parts = rel.parts
            if len(parts) != 4 or parts[-1] != "run.json":
                continue
            model_slug, env_id, exp_id, _ = parts
        except ValueError:
            continue

        try:
            d = json.loads(run_json.read_text())
        except (json.JSONDecodeError, OSError):
            rows.append({
                "model": model_slug, "env": env_id, "exp_id": exp_id,
                "status": "unparseable", "final_score": None,
                "held_out_mean": None, "n_submits": None, "cost_usd": None,
                "path": str(run_json),
            })
            continue

        o = d.get("outcome", {})
        aux = o.get("auxiliary", {})
        agent_meta = d.get("agent_metadata", {})
        rows.append({
            "model": model_slug,
            "env": env_id,
            "exp_id": exp_id,
            "status": o.get("status"),
            "final_score": o.get("final_score"),
            "held_out_mean": o.get("held_out_mean_return"),
            "n_submits": aux.get("n_submits"),
            "cost_usd": agent_meta.get("total_cost_usd"),
            "path": str(run_json),
        })
    return rows


def scan_unfinished(runs_root: Path) -> list[dict[str, Any]]:
    """Find run dirs that started but don't have run.json (aborted / OOM /
    still running). Useful to surface "what's still pending"."""
    rows: list[dict[str, Any]] = []
    if not runs_root.exists():
        return rows
    # Find dirs matching runs/<model>/<env>/<exp-id>/ that have workspace
    # (sign of init) but no run.json.
    for ws in runs_root.glob("*/*/*/workspace"):
        run_json = ws.parent / "run.json"
        if run_json.exists():
            continue
        try:
            rel = ws.parent.relative_to(runs_root)
            model_slug, env_id, exp_id = rel.parts
        except ValueError:
            continue
        rows.append({
            "model": model_slug, "env": env_id, "exp_id": exp_id,
            "status": "no run.json (unfinished / aborted)",
            "final_score": None, "held_out_mean": None,
            "n_submits": None, "cost_usd": None,
            "path": str(ws.parent),
        })
    return rows


def filter_rows(
    rows: list[dict[str, Any]],
    *, model: list[str] | None = None,
    env: list[str] | None = None,
    exp_id: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Substring filter on model/env/exp_id (any-match per filter list)."""
    out = rows
    if model:
        out = [r for r in out if any(m in r["model"] for m in model)]
    if env:
        out = [r for r in out if any(e in r["env"] for e in env)]
    if exp_id:
        out = [r for r in out if any(x in r["exp_id"] for x in exp_id)]
    return out


def _fmt_score(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.2f}"
    return str(v)


def _fmt_int(v: Any) -> str:
    if v is None:
        return "—"
    return str(int(v))


def render_table(rows: list[dict[str, Any]], *, fmt: str = "table") -> str:
    """Render rows as table / markdown / csv / json."""
    if not rows:
        return "(no runs found)"

    cols = ["model", "exp_id", "env", "status", "final_score", "held_out_mean", "n_submits", "cost_usd"]

    if fmt == "json":
        return json.dumps(rows, indent=2)

    if fmt == "csv":
        import io
        buf = io.StringIO()
        w = csvmod.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
        return buf.getvalue()

    if fmt == "markdown":
        out = "| " + " | ".join(cols) + " |\n"
        out += "|" + "|".join(["---"] * len(cols)) + "|\n"
        for r in rows:
            cells = [
                r.get("model", ""),
                r.get("exp_id", ""),
                r.get("env", ""),
                r.get("status", "") or "",
                _fmt_score(r.get("final_score")),
                _fmt_score(r.get("held_out_mean")),
                _fmt_int(r.get("n_submits")),
                _fmt_score(r.get("cost_usd")),
            ]
            out += "| " + " | ".join(cells) + " |\n"
        return out

    # Default: aligned text table
    widths: dict[str, int] = {c: len(c) for c in cols}
    formatted_rows: list[dict[str, str]] = []
    for r in rows:
        fr = {
            "model": str(r.get("model", "")),
            "exp_id": str(r.get("exp_id", "")),
            "env": str(r.get("env", "")),
            "status": str(r.get("status", "") or ""),
            "final_score": _fmt_score(r.get("final_score")),
            "held_out_mean": _fmt_score(r.get("held_out_mean")),
            "n_submits": _fmt_int(r.get("n_submits")),
            "cost_usd": _fmt_score(r.get("cost_usd")),
        }
        formatted_rows.append(fr)
        for c in cols:
            widths[c] = max(widths[c], len(fr[c]))

    line = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    out_text = line + "\n" + "-" * len(line) + "\n"
    for fr in formatted_rows:
        out_text += "  ".join(f"{fr[c]:<{widths[c]}}" for c in cols) + "\n"
    return out_text


def pivot_table(
    rows: list[dict[str, Any]],
    *, pivot_dim: str,
    fmt: str = "markdown",
) -> str:
    """Build a pivot of final_score where one axis = pivot_dim (env or exp_id)
    and the other = the complement.

    Cells = final_score (or "—" if missing). Useful for cross-experiment
    or cross-env comparison.
    """
    assert pivot_dim in ("env", "exp_id"), f"pivot_dim must be 'env' or 'exp_id'; got {pivot_dim}"
    other = "exp_id" if pivot_dim == "env" else "env"

    cell: dict[tuple[str, str], float] = {}
    pivot_keys: set[str] = set()
    other_keys: set[str] = set()
    for r in rows:
        if r.get("status") != "completed":
            continue
        p = r.get(pivot_dim)
        o = r.get(other)
        s = r.get("final_score")
        if p is None or o is None or s is None:
            continue
        cell[(o, p)] = float(s)
        pivot_keys.add(p)
        other_keys.add(o)

    pivot_list = sorted(pivot_keys)
    other_list = sorted(other_keys)

    if fmt == "csv":
        import io
        buf = io.StringIO()
        w = csvmod.writer(buf)
        w.writerow([other] + pivot_list)
        for o in other_list:
            row = [o] + [_fmt_score(cell.get((o, p))) for p in pivot_list]
            w.writerow(row)
        return buf.getvalue()

    # markdown / table
    header = "| " + other + " | " + " | ".join(pivot_list) + " |\n"
    header += "|" + "|".join(["---"] * (1 + len(pivot_list))) + "|\n"
    body = ""
    for o in other_list:
        cells = [o] + [_fmt_score(cell.get((o, p))) for p in pivot_list]
        body += "| " + " | ".join(cells) + " |\n"

    # mean across the "other" dim per pivot column
    mean_row = "| **mean** "
    for p in pivot_list:
        vals = [cell[(o, p)] for o in other_list if (o, p) in cell]
        m = (sum(vals) / len(vals)) if vals else None
        mean_row += "| " + (f"**{m:.2f}**" if m is not None else "—") + " "
    mean_row += "|\n"

    return header + body + mean_row


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--runs-root", type=Path,
                   default=_REPO_ROOT / "runs",
                   help="root directory containing model/env/exp-id/ dirs")
    p.add_argument("--model", nargs="+", default=None,
                   help="filter by model slug substring (e.g. claude-code-auto)")
    p.add_argument("--env", nargs="+", default=None,
                   help="filter by env id substring (e.g. pendulum)")
    p.add_argument("--exp-id", nargs="+", default=None,
                   help="filter by exp-id substring (e.g. v1paper-sonnet)")
    p.add_argument("--format", choices=["table", "markdown", "csv", "json"],
                   default="table",
                   help="output format (default: table)")
    p.add_argument("--pivot", choices=["env", "exp_id"], default=None,
                   help="if set, pivot final_score on this axis "
                        "(rows=other axis, columns=pivot)")
    p.add_argument("--include-errors", action="store_true",
                   help="also list run dirs missing run.json (unfinished / aborted)")
    args = p.parse_args()

    rows = scan_runs(args.runs_root)
    if args.include_errors:
        rows.extend(scan_unfinished(args.runs_root))

    rows = filter_rows(rows, model=args.model, env=args.env, exp_id=args.exp_id)
    rows.sort(key=lambda r: (r["model"], r["exp_id"], r["env"]))

    if args.pivot:
        print(pivot_table(rows, pivot_dim=args.pivot,
                          fmt="csv" if args.format == "csv" else "markdown"))
    else:
        print(render_table(rows, fmt=args.format))

    # Stderr summary so it doesn't pollute pipes
    completed = [r for r in rows if r.get("status") == "completed"]
    errored = [r for r in rows if r.get("status") and r["status"] != "completed"]
    print(f"\n# {len(rows)} rows ({len(completed)} completed, {len(errored)} other)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
