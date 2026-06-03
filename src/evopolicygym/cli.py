"""Command-line entry point for local EvoPolicyGym runs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from .agent import Claude, Codex, Command, Harness, Kimi, Loop
from .check import check as check_run
from .check import check_env
from .check.task import REQUIRED_SECTIONS, check_task_text
from .config import Agent, Server, Spec, load, overlay
from .data import make as make_data
from .envs import manifest as env_manifest
from .envs import registry
from .envs.discover import discover as discover_envs
from .envs.discover import write_json as write_env_json
from .envs.discover import write_markdown as write_env_markdown
from .envs.gym.dynamic import discover as discover_bulk_gym
from .host import Drive, Trial, local
from .layout import root as run_root
from .suite import Result as SuiteResult
from .suite import Suite


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "suite":
        return _suite(args)
    if args.cmd == "data":
        if args.data_cmd == "make":
            return _data_make(args)
        parser.print_help()
        return 2
    if args.cmd == "check-envs":
        return _check_envs(args)
    if args.cmd == "_check-one-env":
        return _check_one_env(args)
    if args.cmd == "discover-envs":
        return _discover_envs(args)
    parser.print_help()
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evopolicygym")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run one local benchmark session")
    run.add_argument("--config", type=Path, help="JSON or TOML run configuration")
    run.add_argument("--env", help="environment name")
    run.add_argument("--bulk", action="store_true", help="enable dynamic Gymnasium bulk env names")
    run.add_argument("--root", type=Path, help="run output directory")
    run.add_argument("--runs", type=Path, help="runs base directory for model/env/exp-id layout")
    run.add_argument("--data", type=Path, help="case split directory with train/valid/heldout JSON")
    run.add_argument("--key", help="run key; defaults to root name")
    run.add_argument("--model", help="agent/model label")
    run.add_argument("--exp", help="experiment label")
    run.add_argument("--exp-id", dest="exp_id", help="experiment id used in runs/model/env/exp-id layout")
    run.add_argument("--budget", type=_nonnegative, help="episode budget")
    run.add_argument("--minimum", type=_positive, help="minimum episodes per submit")
    run.add_argument("--maximum", type=_positive, help="maximum episodes per submit")
    run.add_argument("--valid-size", type=_nonnegative, help="validation pool size override")
    run.add_argument("--final-size", type=_nonnegative, help="final pool size override")
    run.add_argument("--limit", type=_positive, help="agent turn guard")
    run.add_argument("--retries", type=_nonnegative, help="agent harness retry count")
    run.add_argument("--retry-backoff", type=_nonnegative_float, help="agent retry base backoff seconds")
    run.add_argument("--bind", help="server bind address")
    run.add_argument("--port", type=_nonnegative, help="server port; 0 chooses a free port")
    run.add_argument(
        "--agent",
        choices=("command", "codex", "claude", "kimi"),
        help="agent adapter",
    )
    run.add_argument("--agent-name", help="agent log/session name")
    run.add_argument("argv", nargs=argparse.REMAINDER, help="agent command after --")

    suite = sub.add_parser("suite", help="run a serial benchmark suite")
    suite.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON or TOML suite configuration",
    )

    data = sub.add_parser("data", help="manage external case split data")
    data_sub = data.add_subparsers(dest="data_cmd")
    make = data_sub.add_parser("make", help="write seed-backed train/valid/heldout splits")
    make.add_argument("--env", required=True, help="environment name")
    make.add_argument("--bulk", action="store_true", help="enable dynamic Gymnasium bulk env names")
    make.add_argument("--root", type=Path, required=True, help="output data directory")
    make.add_argument("--seed", type=int, default=0, help="master seed for deterministic splits")
    make.add_argument("--train-size", type=_positive, help="number of train cases")
    make.add_argument("--valid-size", type=_positive, help="number of validation cases")
    make.add_argument("--heldout-size", type=_positive, help="number of held-out cases")
    make.add_argument("--force", action="store_true", help="overwrite existing split files")

    check_envs = sub.add_parser("check-envs", help="check registered environments against the manifest")
    check_envs.add_argument("--env", action="append", help="filter by EvoPolicyGym name or upstream id")
    check_envs.add_argument("--bulk", action="store_true", help="register installed Gymnasium ids as gymnasium/<id>")
    check_envs.add_argument("--source", type=Path, help="read discovered-envs JSON as the bulk planning source")
    check_envs.add_argument("--family", action="append", help="filter by manifest/discovery family name")
    check_envs.add_argument("--discover", action="store_true", help="include installed L0 discovery entries")
    check_envs.add_argument("--min-level", default="L0", help="minimum support level to report, e.g. L0 or L2")
    check_envs.add_argument(
        "--isolate",
        action="store_true",
        help="check each registered environment in a subprocess",
    )
    check_envs.add_argument(
        "--timeout",
        type=_nonnegative_float,
        default=30.0,
        help="seconds allowed for each isolated environment check; 0 disables the limit",
    )
    check_envs.add_argument(
        "--jobs",
        type=_positive,
        default=1,
        help="maximum concurrent isolated environment checks",
    )

    one = sub.add_parser("_check-one-env", help=argparse.SUPPRESS)
    one.add_argument("--env", required=True, help=argparse.SUPPRESS)
    one.add_argument("--bulk", action="store_true", help=argparse.SUPPRESS)

    discover = sub.add_parser("discover-envs", help="discover installed environment registries")
    discover.add_argument("--output", type=Path, help="write JSON report")
    discover.add_argument("--markdown", type=Path, help="write Markdown report")
    return parser


def _run(args: argparse.Namespace) -> int:
    spec = _spec(args)
    trial = _trial(spec, bulk=args.bulk)
    body = _summary(spec, trial)
    print(json.dumps(body, sort_keys=True))
    return 0 if trial.done else 1


def _suite(args: argparse.Namespace) -> int:
    try:
        suite = Suite.load(args.config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    results = _suite_results(suite)
    suite.write(results)
    body = suite.report(results)
    print(json.dumps(body, sort_keys=True))
    return 0 if body["done"] else 1


def _discover_envs(args: argparse.Namespace) -> int:
    report = discover_envs()
    if args.output:
        write_env_json(report, args.output)
    if args.markdown:
        write_env_markdown(report, args.markdown)
    if not args.output and not args.markdown:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(json.dumps({"total": report.total, "families": len(report.families)}, sort_keys=True))
    return 0


def _data_make(args: argparse.Namespace) -> int:
    try:
        env = registry(bulk=args.bulk, filters=(args.env,)).get(args.env)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        corpus = make_data(
            args.root,
            env,
            seed=args.seed,
            train_size=args.train_size,
            valid_size=args.valid_size,
            heldout_size=args.heldout_size,
            overwrite=args.force,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    body = {
        "env": env.task.name,
        "root": str(corpus.root),
        "train": corpus.train.pool.size,
        "valid": corpus.valid.pool.size,
        "heldout": corpus.final.pool.size,
        "versions": corpus.versions(),
    }
    print(json.dumps(body, sort_keys=True))
    return 0


def _check_envs(args: argparse.Namespace) -> int:
    try:
        minimum = env_manifest.Level.parse(args.min_level)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    filters = set(args.env or ())
    families = set(args.family or ())
    source_ids = _source_ids(args.source, families) if args.source is not None else set()
    bulk_filters = filters or source_ids
    if args.source is not None and not bulk_filters:
        bulk_filters = {"__evopolicygym_no_source_match__"}
    catalog = None if args.isolate else registry(bulk=args.bulk, filters=tuple(bulk_filters))
    registered = set() if catalog is None else set(catalog.list())
    entries = list(env_manifest.entries())
    if args.bulk:
        entries.extend(env_manifest.from_bulk(discover_bulk_gym(bulk_filters)))
    if args.discover:
        entries.extend(env_manifest.from_discovery(discover_envs()))

    if args.source is not None and source_ids and not filters:
        names = {f"gymnasium/{env_id}" for env_id in source_ids}
        entries = [entry for entry in entries if entry.upstream in source_ids or entry.name in names]

    if catalog is not None:
        known = {entry.name for entry in entries}
        for name in sorted(registered - known):
            entries.append(
                env_manifest.Entry(
                    name=name,
                    level=env_manifest.Level.smoke,
                    family="Unclassified",
                    adapter="registered",
                    notes="registered environment missing from manifest",
                )
            )

    if filters:
        entries = [entry for entry in entries if entry.name in filters or entry.upstream in filters]
        if not entries:
            raise SystemExit(f"no environment matched: {', '.join(sorted(filters))}")
    if families:
        entries = [entry for entry in entries if entry.family in families]
        if not entries:
            raise SystemExit(f"no family matched: {', '.join(sorted(families))}")

    entries = [entry for entry in entries if entry.level >= minimum]
    rows = (
        _env_statuses_isolated(entries, args.bulk, args.timeout, args.jobs)
        if args.isolate
        else [_env_status(entry, catalog) for entry in entries]
    )
    body = {
        "total": len(rows),
        "registered": sum(1 for row in rows if row["registered"]),
        "checked": sum(1 for row in rows if row["checked"]),
        "ok": sum(1 for row in rows if row["ok"] is True),
        "failed": sum(1 for row in rows if row["ok"] is False),
        "missing": sum(1 for row in rows if not row["registered"]),
        "by_level": _by_level(rows),
        "envs": rows,
    }
    print(json.dumps(body, sort_keys=True))
    return 1 if body["failed"] else 0


def _source_ids(path: Path, families: set[str]) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"failed to read discovery source {path}: {exc}") from exc
    rows = data.get("families") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise SystemExit(f"discovery source {path} must contain a families array")

    ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        family = row.get("name")
        source = row.get("source")
        values = row.get("ids")
        if families and family not in families:
            continue
        if source != "gymnasium.registry" or not isinstance(values, list):
            continue
        ids.update(item for item in values if isinstance(item, str))
    return ids


def _env_status(entry: env_manifest.Entry, catalog) -> dict[str, object]:
    row = entry.as_dict()
    registered = entry.name in catalog.list()
    row["registered"] = registered
    row["checked"] = False
    row["ok"] = None
    row["issues"] = []
    if not registered:
        return row

    row["checked"] = True
    try:
        env = catalog.get(entry.name)
        report = check_env(env)
    except Exception as exc:  # noqa: BLE001 - status reports must not crash the suite.
        row["ok"] = False
        row["issues"] = [
            {
                "code": "check_exception",
                "path": entry.name,
                "message": f"{type(exc).__name__}: {exc}",
            }
        ]
        return row
    issues = list(report.issues)
    if entry.level >= env_manifest.Level.tasked:
        issues.extend(
            check_task_text(
                env.text,
                path=entry.name,
                required=REQUIRED_SECTIONS,
            )
        )
    row["ok"] = not issues
    row["issues"] = [
        {"code": issue.code, "path": issue.path, "message": issue.message}
        for issue in issues
    ]
    return row


def _env_status_isolated(
    entry: env_manifest.Entry,
    bulk: bool,
    timeout: float,
) -> dict[str, object]:
    row = entry.as_dict()
    row["registered"] = False
    row["checked"] = False
    row["ok"] = None
    row["issues"] = []
    if entry.level == env_manifest.Level.catalogued:
        return row

    cmd = [sys.executable, "-m", "evopolicygym.cli", "_check-one-env", "--env", entry.name]
    if bulk:
        cmd.append("--bulk")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=None if timeout <= 0 else timeout,
        )
    except subprocess.TimeoutExpired as exc:
        row["registered"] = True
        row["checked"] = True
        row["ok"] = False
        row["issues"] = [
            {
                "code": "check_timeout",
                "path": entry.name,
                "message": f"isolated check timed out after {timeout:g}s",
            }
        ]
        if exc.stdout:
            row["stdout"] = _tail(str(exc.stdout))
        if exc.stderr:
            row["stderr"] = _tail(str(exc.stderr))
        return row

    if result.returncode != 0:
        row["registered"] = True
        row["checked"] = True
        row["ok"] = False
        row["issues"] = [
            {
                "code": "check_crash",
                "path": entry.name,
                "message": f"isolated check exited with status {result.returncode}",
            }
        ]
        if result.stdout:
            row["stdout"] = _tail(result.stdout)
        if result.stderr:
            row["stderr"] = _tail(result.stderr)
        return row

    try:
        child = _last_json(result.stdout)
    except ValueError as exc:
        row["registered"] = True
        row["checked"] = True
        row["ok"] = False
        row["issues"] = [
            {
                "code": "check_protocol",
                "path": entry.name,
                "message": str(exc),
            }
        ]
        if result.stdout:
            row["stdout"] = _tail(result.stdout)
        if result.stderr:
            row["stderr"] = _tail(result.stderr)
        return row

    for key in ("registered", "checked", "ok", "issues"):
        row[key] = child.get(key)
    if result.stderr:
        row["stderr"] = _tail(result.stderr)
    return row


def _env_statuses_isolated(
    entries: list[env_manifest.Entry],
    bulk: bool,
    timeout: float,
    jobs: int,
) -> list[dict[str, object]]:
    if jobs <= 1 or len(entries) <= 1:
        return [_env_status_isolated(entry, bulk, timeout) for entry in entries]

    rows: list[dict[str, object] | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_env_status_isolated, entry, bulk, timeout): index
            for index, entry in enumerate(entries)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    return [row for row in rows if row is not None]


def _check_one_env(args: argparse.Namespace) -> int:
    body: dict[str, object] = {
        "registered": False,
        "checked": False,
        "ok": None,
        "issues": [],
    }
    try:
        report = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                catalog = registry(bulk=args.bulk, filters=(args.env,))
                if args.env not in set(catalog.list()):
                    body["issues"] = [
                        {
                            "code": "not_registered",
                            "path": args.env,
                            "message": "environment was not registered in the filtered catalog",
                        }
                    ]
                else:
                    env = catalog.get(args.env)
                    report = check_env(env)
        if report is not None:
            body["registered"] = True
            body["checked"] = True
            body["ok"] = report.ok
            body["issues"] = [
                {"code": issue.code, "path": issue.path, "message": issue.message}
                for issue in report.issues
            ]
    except Exception as exc:  # noqa: BLE001 - isolated check must report exceptions.
        body["registered"] = True
        body["checked"] = True
        body["ok"] = False
        body["issues"] = [
            {
                "code": "check_exception",
                "path": args.env,
                "message": f"{type(exc).__name__}: {exc}",
            }
        ]
    print(json.dumps(body, sort_keys=True))
    return 0


def _last_json(value: str) -> dict[str, object]:
    for line in reversed(value.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict):
            return body
    raise ValueError("isolated check did not emit a JSON object")


def _tail(value: str, *, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _by_level(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {level.code: 0 for level in env_manifest.Level}
    for row in rows:
        level = row.get("level")
        if isinstance(level, str):
            counts[level] = counts.get(level, 0) + 1
    return counts


def _suite_results(suite: Suite) -> list[SuiteResult]:
    if suite.concurrency == 1:
        return [_job(job) for job in suite.jobs]

    results: list[SuiteResult | None] = [None] * len(suite.jobs)
    with ThreadPoolExecutor(max_workers=suite.concurrency) as executor:
        futures = {executor.submit(_job, job): job.index for job in suite.jobs}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    return [result for result in results if result is not None]


def _job(job) -> SuiteResult:
    try:
        trial = _trial(job.spec)
        result = SuiteResult.from_summary(job, _summary(job.spec, trial))
    except Exception as exc:
        return SuiteResult.from_error(job, exc)
    return _checked(result)


def _checked(result: SuiteResult) -> SuiteResult:
    try:
        report = check_run(result.root)
    except Exception as exc:
        return result.checked(False, (f"checker:{type(exc).__name__}:{exc}",))
    issues = tuple(f"{issue.code}:{issue.path}:{issue.message}" for issue in report.issues)
    return result.checked(report.ok, issues)


def _trial(spec: Spec, *, bulk: bool = False) -> Trial:
    try:
        env = registry(bulk=bulk, filters=(spec.env,)).get(spec.env)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc

    host = local(
        spec.root,
        env,
        key=spec.run_key,
        model=spec.model,
        exp=spec.exp,
        budget=spec.budget,
        data=spec.data,
        minimum=spec.minimum,
        maximum=spec.maximum,
        valid_size=spec.valid_size,
        final_size=spec.final_size,
    )
    loop = Loop(
        _harness(spec.agent),
        limit=spec.agent.limit,
        retries=spec.agent.retries,
        backoff=spec.agent.retry_backoff,
    )
    return Drive(loop, bind=spec.server.bind, port=spec.server.port).run(host)


def _summary(spec: Spec, trial: Trial) -> dict[str, object]:
    return {
        "done": trial.done,
        "reason": trial.transcript.reason,
        "root": str(spec.root),
        "run": trial.host.run.key,
        "session": trial.transcript.session,
        "submits": trial.host.service.submits,
    }


def _spec(args: argparse.Namespace) -> Spec:
    if args.config is None:
        if args.root is None and args.runs is None:
            raise SystemExit("--root or --runs is required without --config")
        if args.budget is None:
            raise SystemExit("--budget is required without --config")
        spec = Spec(
            env=args.env or "toy",
            root=args.root,
            runs=args.runs,
            model=args.model or "agent",
            exp=args.exp_id or args.exp or "default",
            budget=args.budget,
            data=args.data,
        )
    else:
        try:
            spec = load(args.config)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc

    spec = overlay(
        spec,
        env=args.env,
        root=args.root,
        runs=args.runs,
        data=args.data,
        key=args.key,
        model=args.model,
        exp=args.exp_id or args.exp,
        budget=args.budget,
        minimum=args.minimum,
        maximum=args.maximum,
        valid_size=args.valid_size,
        final_size=args.final_size,
    )
    if spec.runs is not None and args.root is None:
        spec = replace(
            spec,
            root=run_root(spec.runs, model=spec.model, env=spec.env, exp=spec.exp),
        )
    agent = _agent(spec.agent, args)
    server = _server(spec.server, args)
    spec = replace(spec, agent=agent, server=server)
    if spec.agent.kind == "command" and not spec.agent.argv:
        raise SystemExit("agent command is required in config or after --")
    return spec


def _agent(agent: Agent, args: argparse.Namespace) -> Agent:
    argv = _command(args.argv) if args.argv else agent.argv
    kind = args.agent or agent.kind
    binary = agent.binary
    passthrough = agent.args
    if kind in {"codex", "claude", "kimi"} and argv:
        binary = argv[0]
        passthrough = argv[1:]
    return replace(
        agent,
        kind=kind,
        argv=argv,
        name=args.agent_name or agent.name,
        limit=args.limit or agent.limit,
        retries=args.retries if args.retries is not None else agent.retries,
        retry_backoff=(
            args.retry_backoff if args.retry_backoff is not None else agent.retry_backoff
        ),
        binary=binary,
        args=passthrough,
    )


def _server(server: Server, args: argparse.Namespace) -> Server:
    return replace(
        server,
        bind=args.bind or server.bind,
        port=args.port if args.port is not None else server.port,
    )


def _harness(agent: Agent) -> Harness:
    if agent.kind == "command":
        return Command(agent.argv, name=agent.name)
    if agent.kind == "codex":
        return Codex(
            binary=agent.binary or "codex",
            model=agent.model,
            sandbox=agent.sandbox,
            approval=agent.approval,
            bypass=agent.bypass,
            args=agent.args,
            name=agent.name if agent.name != "agent" else "codex",
        )
    if agent.kind == "claude":
        return Claude(
            binary=agent.binary or "claude",
            model=agent.model,
            permission=agent.permission,
            tools=agent.tools or ("Bash", "Read", "Edit", "Write", "Glob", "Grep"),
            args=agent.args,
            name=agent.name if agent.name != "agent" else "claude",
        )
    if agent.kind == "kimi":
        return Kimi(
            binary=agent.binary or "kimi",
            model=agent.model or "kimi-k2",
            args=agent.args,
            name=agent.name if agent.name != "agent" else "kimi",
        )
    raise SystemExit(f"unsupported agent kind: {agent.kind}")


def _command(argv: Sequence[str]) -> tuple[str, ...]:
    values = tuple(argv)
    if values and values[0] == "--":
        values = values[1:]
    if not values:
        raise SystemExit("agent command is required after --")
    return values


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
