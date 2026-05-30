"""Parallel evaluation matrix runner.

Spawn N concurrent ``hlbench agent`` processes across a set of envs,
wait for all to complete, then aggregate ``run.json`` +
``logs/harness_runner.json`` into a single summary table (per-env
``final_score`` / ``held_out_mean`` / ``turns`` / cost / wall time).

Each run is isolated:
  - its own ephemeral HTTP port (per ``hlbench agent --port 0`` default)
  - its own run dir ``<runs-root>/<model-slug>/<env>/<exp-id>/``
  - its own agent session label (claude UUID or codex-scraped id)
  - its own stdout/stderr log at ``<runs-root>/_matrix_logs/<exp-id>__<env>.log``

So no coordination code is needed — each subprocess fully isolates
from its siblings.

Usage::

    .venv/bin/python scripts/run_matrix.py
        # All registered envs, budget=32, max-turns=8, sonnet — small probe.

    .venv/bin/python scripts/run_matrix.py \\
        --envs pendulum acrobot mountain_car_continuous bipedal_walker \\
        --budget 256 --max-turns 24 --model sonnet
        # Full calibration round on 4 envs (cost ~$20-80, wall ~60-90 min).

    .venv/bin/python scripts/run_matrix.py \\
        --envs pendulum --budget 16 --max-turns 4 --model haiku
        # Cheap single-env probe to validate the pipeline (~$0.50).

    .venv/bin/python scripts/run_matrix.py \\
        --backend codex --model gpt-5-codex --model-slug codex-auto \\
        --envs pendulum --budget 16 --max-turns 4
        # Same matrix shape but driven by OpenAI Codex CLI.

Exit code: 0 if every env completed with ``run.json:outcome.status ==
"completed"``; 1 otherwise. Even on partial failure, the summary table
is still printed for whatever did finish.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Sit at repo root so we can resolve the bench's local .venv/bin/hlbench
# without needing the user to source / activate anything.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HLBENCH = _REPO_ROOT / ".venv" / "bin" / "hlbench"


def _registered_envs() -> list[str]:
    """All envs currently registered with the registry. Auto-discovers
    so the script doesn't go stale when new envs are added."""
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    import hlbench.envs  # noqa: F401  side effect: registration
    from hlbench.envs.registry import list_envs
    return list_envs()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],  # first paragraph only
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- matrix axis: envs ----
    p.add_argument(
        "--envs", nargs="+", default=None,
        help="env ids to run in parallel (default: all registered envs)",
    )

    # ---- per-run knobs (apply to every env in the matrix) ----
    p.add_argument("--budget", type=int, default=32,
                   help="episode_budget per env (default: 32)")
    p.add_argument("--max-turns", type=int, default=8,
                   help="harness max_turns per env (default: 8)")
    p.add_argument(
        "--backend", default="claude", choices=("claude", "codex"),
        help="agent CLI backend (default: claude). Forwarded to "
             "`hlbench agent --backend`.",
    )
    p.add_argument("--model", default="sonnet",
                   help="--model passed to the chosen backend "
                        "(default: sonnet for claude, set --model gpt-5-codex "
                        "or similar for codex)")
    p.add_argument("--model-slug", default="claude-code-auto",
                   help="run.json:model slug; also the runs-root subdir")
    p.add_argument("--turn-timeout", type=int, default=900,
                   help="seconds per agent turn (default: 900)")
    p.add_argument("--codex-binary", default=None,
                   help="path to codex binary (only used when --backend codex; "
                        "default: 'codex' on PATH)")

    # ---- run-dir layout ----
    p.add_argument("--runs-root", default="./runs",
                   help="root for runs/<model>/<env>/<exp-id>/ (default: ./runs)")
    p.add_argument(
        "--exp-id", default=None,
        help="exp-id shared across all envs in this matrix "
             "(default: matrix-YYYYmmdd-HHMMSS)",
    )

    # ---- scheduling ----
    p.add_argument(
        "--max-parallel", type=int, default=0,
        help="max concurrent envs in flight (default: 0 = all at once)",
    )

    # ---- plumbing ----
    p.add_argument(
        "--hlbench-bin", default=str(_DEFAULT_HLBENCH),
        help=f"path to hlbench binary (default: {_DEFAULT_HLBENCH})",
    )
    return p.parse_args()


def _build_cmd(args: argparse.Namespace, env_id: str, exp_id: str) -> list[str]:
    """The exact ``hlbench agent`` invocation for one env."""
    cmd = [
        args.hlbench_bin, "agent",
        "--backend", args.backend,
        "--env", env_id,
        "--budget", str(args.budget),
        "--max-turns", str(args.max_turns),
        "--model", args.model,
        "--model-slug", args.model_slug,
        "--turn-timeout", str(args.turn_timeout),
        "--runs-root", args.runs_root,
        "--exp-id", exp_id,
    ]
    if args.backend == "codex" and args.codex_binary:
        cmd.extend(["--codex-binary", args.codex_binary])
    return cmd


def _run_one_env(
    env_id: str,
    args: argparse.Namespace,
    exp_id: str,
    log_dir: Path,
) -> tuple[str, int, float, Path]:
    """Spawn ``hlbench agent`` for one env. Capture combined stdout/stderr
    to a per-env log file; return when the subprocess exits."""
    log_path = log_dir / f"{exp_id}__{env_id}.log"
    cmd = _build_cmd(args, env_id, exp_id)
    started = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"# cmd: {' '.join(cmd)}\n# started: {datetime.now().isoformat()}\n\n")
        log.flush()
        completed = subprocess.run(  # noqa: S603
            cmd, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.monotonic() - started
    return env_id, completed.returncode, elapsed, log_path


def _read_run_summary(
    runs_root: str, model_slug: str, env_id: str, exp_id: str,
) -> dict | None:
    """Load run.json for one run; None if missing (run failed before finalize)."""
    p = Path(runs_root) / model_slug / env_id / exp_id / "run.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        assert isinstance(data, dict)
        return data
    except (json.JSONDecodeError, AssertionError):
        return None


def _read_harness_runner(
    runs_root: str, model_slug: str, env_id: str, exp_id: str,
) -> dict | None:
    """Load harness_runner.json (cost / turns / wall) for one run."""
    p = (Path(runs_root) / model_slug / env_id / exp_id
         / "logs" / "harness_runner.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        assert isinstance(data, dict)
        return data
    except (json.JSONDecodeError, AssertionError):
        return None


def _print_table(rows: list[list[str]]) -> None:
    """Pretty-print a 2-D list of strings as an aligned text table."""
    if not rows:
        return
    n_cols = len(rows[0])
    widths = [max(len(r[i]) for r in rows) for i in range(n_cols)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*rows[0]))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows[1:]:
        print(fmt.format(*row))


def main() -> int:
    args = parse_args()
    envs = args.envs if args.envs is not None else _registered_envs()
    if not envs:
        print("error: no envs to run", file=sys.stderr)
        return 2

    if not Path(args.hlbench_bin).is_file():
        print(f"error: hlbench binary not found at {args.hlbench_bin}", file=sys.stderr)
        print("       (override with --hlbench-bin /path/to/hlbench)", file=sys.stderr)
        return 2

    exp_id = args.exp_id or f"matrix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = Path(args.runs_root) / "_matrix_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    max_workers = args.max_parallel if args.max_parallel > 0 else len(envs)

    print("=" * 70)
    print(f"hlbench eval matrix — {len(envs)} env(s), up to {max_workers} concurrent")
    print("=" * 70)
    print(f"  exp_id:        {exp_id}")
    print(f"  budget/env:    {args.budget}")
    print(f"  max-turns/env: {args.max_turns}")
    print(f"  model:         {args.model}")
    print(f"  runs-root:     {args.runs_root}")
    print(f"  log dir:       {log_dir}")
    print()
    print("Envs:")
    for env in envs:
        print(f"  - {env}")
    print()
    print("Per-env progress: tail one of the log files, e.g.")
    print(f"  tail -f {log_dir}/{exp_id}__{envs[0]}.log")
    print()

    matrix_started = time.monotonic()
    completion: dict[str, tuple[int, float, Path]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_one_env, env, args, exp_id, log_dir): env
            for env in envs
        }
        for fut in as_completed(futures):
            env_id = futures[fut]
            try:
                env_id, rc, elapsed, log_path = fut.result()
            except Exception as e:  # pragma: no cover (defensive)
                print(f"[FAIL] {env_id} — runner crashed: {e}", file=sys.stderr)
                completion[env_id] = (1, 0.0, log_dir / f"{exp_id}__{env_id}.log")
                continue
            status = "ok" if rc == 0 else f"exit={rc}"
            print(f"[done] {env_id} ({status}) {elapsed:.0f}s")
            completion[env_id] = (rc, elapsed, log_path)

    total_wall = time.monotonic() - matrix_started
    print()
    print("=" * 70)
    print(f"Matrix complete in {total_wall:.0f}s total wall time.")
    print("=" * 70)
    print()

    # Aggregate run.json + harness_runner.json into a table.
    table: list[list[str]] = [[
        "env", "status", "final_score", "held_out_mean",
        "turns", "termination", "cost($)", "wall_s",
    ]]
    grand_cost = 0.0
    n_ok = 0
    for env_id in envs:
        run_data = _read_run_summary(args.runs_root, args.model_slug, env_id, exp_id)
        runner_data = _read_harness_runner(args.runs_root, args.model_slug, env_id, exp_id)

        outcome = (run_data or {}).get("outcome", {})
        status = outcome.get("status", "no_run_json")
        score = outcome.get("final_score")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        hmean = outcome.get("held_out_mean_return")
        hmean_s = f"{hmean:.2f}" if isinstance(hmean, (int, float)) else "-"

        if runner_data is not None:
            turns = runner_data.get("n_turns", 0)
            termination = runner_data.get("termination_reason", "-")
            cost = runner_data.get("total_cost_usd", 0.0) or 0.0
            wall = runner_data.get("wall_time_seconds", 0.0) or 0.0
            grand_cost += cost
        else:
            turns = 0
            termination = "-"
            cost = 0.0
            wall = 0.0

        if status == "completed":
            n_ok += 1

        table.append([
            env_id,
            status,
            score_s,
            hmean_s,
            str(turns),
            str(termination),
            f"${cost:.2f}",
            f"{wall:.0f}",
        ])

    _print_table(table)
    print()
    print(f"  envs completed: {n_ok}/{len(envs)}")
    print(f"  total cost:     ${grand_cost:.2f}")
    print(f"  total wall:     {total_wall:.0f}s")
    print()

    failed = [
        env for env, (rc, _, _) in completion.items() if rc != 0
    ] + [
        env for env in envs
        if env not in completion or _read_run_summary(
            args.runs_root, args.model_slug, env, exp_id
        ) is None
    ]
    failed = sorted(set(failed))
    if failed:
        print(f"  failed envs: {failed}", file=sys.stderr)
        print(f"  see logs:    {log_dir}/{exp_id}__<env>.log", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
