"""hlbench CLI entry point. Run ``hlbench --help`` for subcommands.

Workflow::

    hlbench init --env pendulum --dir ./run         # one-time setup
    hlbench serve --workspace ./run                 # in foreground
    # ... in another terminal ...
    hlbench info
    hlbench submit --env-instances 0-3
    hlbench finalize

The CLI is just a thin HTTP client (no shared state with the server).
For programmatic use, drive ``hlbench.core.Server`` directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:8765"


# --------------------------- helpers --------------------------------------


def _parse_env_instances(spec: str) -> list[int]:
    """Accept ``"0-3"`` (range, inclusive), ``"0,2,5"`` (list), or
    a mix: ``"0-3,7,10-12"``."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def _http_request(url: str, *, method: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode("utf-8") if method == "POST" else None
    headers = {"Content-Type": "application/json"} if method == "POST" else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        try:
            err_body = json.loads(body_text)
        except json.JSONDecodeError:
            err_body = {"error": "non_json_response", "raw": body_text}
        print(f"HTTP {e.code}: {err_body.get('message', err_body)}", file=sys.stderr)
        sys.exit(2)
    except urllib.error.URLError as e:
        print(f"connection error: {e.reason}", file=sys.stderr)
        print(f"  is `hlbench serve` running on {url}?", file=sys.stderr)
        sys.exit(2)


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


# --------------------------- subcommands ---------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """Create a fresh workspace. Just calls ``Server(...)`` and exits —
    Server.__init__ stages TASK.md + AGENT.md + system/ + feedback/."""
    from hlbench.core.server import Server  # lazy: only init needs it

    workspace = Path(args.dir)
    server = Server(
        env_id=args.env,
        workspace_dir=workspace,
        model=args.model,
        exp_id=args.exp_id,
    )
    info = server.info()
    print(f"Initialized {args.env} workspace at {workspace.resolve()}")
    print(f"  episode_budget    {info['episode_budget']}")
    print(f"  n_env_instances   {info['env_meta']['n_env_instances']}")
    print(f"  agent_md_hash     {info['agent_md_hash']}")
    print()
    print("Next: drop a Policy at workspace/system/policy.py, then run:")
    print(f"  hlbench serve --workspace {workspace}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Spin up the HTTP server in the foreground. Blocks until Ctrl-C."""
    from hlbench.core.server import Server
    from hlbench.http_server import HlbenchHTTPServer
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    server = Server(
        env_id=args.env,
        workspace_dir=Path(args.workspace),
        model=args.model,
        exp_id=args.exp_id,
    )
    http = HlbenchHTTPServer(server, host=args.host, port=args.port)
    print(f"hlbench server listening on http://{args.host}:{args.port}")
    print(f"  workspace: {Path(args.workspace).resolve()}")
    print(f"  env:       {args.env}")
    print("Ctrl-C to stop.")
    try:
        http.serve_forever_blocking()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    info = _http_request(f"{args.url}/info", method="GET")
    if args.raw:
        _print_json(info)
        return 0
    state = info["state"]
    print(f"env:           {info['env']} (v{info['env_version']})")
    print(f"harness:       {info['harness_version']}")
    print(f"agent_md_hash: {info['agent_md_hash']}")
    print(f"budget:        {state['remaining_budget']} / {info['episode_budget']}")
    print(f"submits:       {state['n_submits']} ({state['n_successful_submits']} ok)")
    print(f"last_submit:   "
          f"#{state['last_submit_index']} → {state['last_submit_status']}")
    print(f"finalized:     {state['is_finalized']}")
    print(f"started_at:    {state['started_at']}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    ids = _parse_env_instances(args.env_instances)
    result = _http_request(
        f"{args.url}/submit", method="POST", body={"env_instances": ids},
    )
    if args.raw:
        _print_json(result)
        return 0
    s = result["summary"]
    print(f"submit #{result['submit_id']}: {result['status']}")
    if result["status"] == "ok":
        print(f"  episodes:        {s['n_episodes']} (ids {s['env_instances']})")
        print(f"  mean_return:     {s['mean_return']:.2f}")
        print(f"  std/min/max:     "
              f"{s['std_return']:.2f} / {s['min_return']:.2f} / {s['max_return']:.2f}")
        print(f"  remaining:       {s['remaining_budget']}")
        if s["timeouts"]:
            print(f"  timeouts at idx: {s['timeouts']}")
        if s["errors"]:
            print(f"  errors at idx:   {s['errors']}")
    else:
        print(f"  remaining: {s.get('remaining_budget')}")
        print(f"  see: workspace/feedback/submit_*/errors.txt")
    return 0 if result["status"] == "ok" else 1


def cmd_finalize(args: argparse.Namespace) -> int:
    result = _http_request(f"{args.url}/finalize", method="POST")
    if args.raw:
        _print_json(result)
        return 0
    print(f"finalize: {result['status']}")
    if result["status"] == "completed":
        print(f"  final_score:        {result['final_score']:.2f}")
        print(f"  held_out_mean:      {result['held_out_mean_return']:.2f}")
        print(f"  held_out_std:       {result['held_out_std_return']:.2f}")
        print(f"  final_submit:       #{result['final_submit_index']}")
        print(f"  run.json:           {result['run_json_path']}")
    else:
        print(f"  error: {result['error']}")
        print(f"  run.json: {result['run_json_path']}")
    return 0 if result["status"] == "completed" else 1


# --------------------------- argparse ------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hlbench",
        description=(
            "hlbench CLI — manage a benchmark run. Use `init` once to "
            "create a workspace, `serve` to expose it over HTTP, then "
            "`info` / `submit` / `finalize` against the running server."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create a fresh workspace")
    p_init.add_argument("--env", required=True, help="registered env id (e.g. 'pendulum')")
    p_init.add_argument("--dir", required=True, help="path to create the workspace at")
    p_init.add_argument("--model", default="unknown", help="recorded as run.json:model")
    p_init.add_argument("--exp-id", default=None, help="recorded as run.json:exp_id")
    p_init.set_defaults(func=cmd_init)

    p_serve = sub.add_parser("serve", help="start HTTP server (foreground)")
    p_serve.add_argument("--workspace", required=True, help="path to existing workspace")
    p_serve.add_argument("--env", required=True, help="env id matching the workspace")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--model", default="unknown")
    p_serve.add_argument("--exp-id", default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_info = sub.add_parser("info", help="GET /info")
    p_info.add_argument("--url", default=DEFAULT_URL)
    p_info.add_argument("--raw", action="store_true", help="print full JSON")
    p_info.set_defaults(func=cmd_info)

    p_sub = sub.add_parser("submit", help="POST /submit")
    p_sub.add_argument(
        "--env-instances", required=True,
        help="env-instance IDs as range '0-7' or list '0,2,5' or mix '0-3,7'",
    )
    p_sub.add_argument("--url", default=DEFAULT_URL)
    p_sub.add_argument("--raw", action="store_true")
    p_sub.set_defaults(func=cmd_submit)

    p_fin = sub.add_parser("finalize", help="POST /finalize")
    p_fin.add_argument("--url", default=DEFAULT_URL)
    p_fin.add_argument("--raw", action="store_true")
    p_fin.set_defaults(func=cmd_finalize)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
