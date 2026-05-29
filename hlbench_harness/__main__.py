"""``hlbench-agent`` — automated evaluation CLI.

Drives one full run end-to-end:

    hlbench-agent --env pendulum --budget 32 --max-turns 12 \\
                  --model sonnet --runs-root ./runs \\
                  --exp-id dogfood-1

Internally:
  1. ``Server(env_id, runs_root, model, exp_id, episode_budget=...)``
  2. ``HlbenchHTTPServer`` on a background thread (port=0 by default
     so parallel runs don't collide).
  3. ``ClaudeAgent`` with a fresh UUID; cwd = workspace.
  4. ``HarnessRunner.run()`` — blocks until the agent finalizes or
     ``max_turns`` is hit.

Logs:
  - ``<run_dir>/logs/agent_turns/turn_NNN.{json,txt,prompt.txt}``
  - ``<run_dir>/logs/harness_runner.json`` (per-turn timeline + final)
  - ``<run_dir>/logs/harness.log`` (existing lifecycle events)
  - ``<run_dir>/run.json`` (the headline; finalize is always called)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hlbench.core.server import Server
from hlbench.http_server import HlbenchHTTPServer
from hlbench_harness.claude_agent import (
    DEFAULT_ALLOWED_TOOLS,
    ClaudeAgent,
    ClaudeAgentConfig,
    find_claude_binary,
    new_session_id,
)
from hlbench_harness.runner import HarnessRunner, RunSummary

log = logging.getLogger("hlbench_harness")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.require_claude and find_claude_binary() is None:
        print(
            "error: 'claude' binary not found on PATH. "
            "Install Claude Code or pass --no-require-claude (test mode).",
            file=sys.stderr,
        )
        return 2

    # 1. Spin up the per-run Server.
    overrides: dict[str, int] = {}
    if args.budget is not None:
        overrides["episode_budget"] = args.budget
    if args.max_episodes_per_submit is not None:
        overrides["max_episodes_per_submit"] = args.max_episodes_per_submit

    server = Server(
        env_id=args.env,
        runs_root=Path(args.runs_root).resolve(),
        model=args.model_slug,
        exp_id=args.exp_id,
        config_overrides=overrides or None,
    )
    log.info(
        "Server initialized: run_dir=%s exp_id=%s",
        server.run_dir, server.exp_id,
    )

    # 2. HTTP server on a background thread.
    http = HlbenchHTTPServer(server, host=args.host, port=args.port)
    http.start()
    try:
        url = f"http://{http.host}:{http.port}"
        log.info("HTTP server live: %s", url)

        # 3. Build the agent.
        session_id = args.session_id or new_session_id()
        agent_log_dir = server.run_dir / "logs" / "agent_turns"
        agent = ClaudeAgent(
            workspace_dir=server.workspace_dir,
            http_url=url,
            log_dir=agent_log_dir,
            config=ClaudeAgentConfig(
                model=args.model,
                permission_mode=args.permission_mode,
                allowed_tools=tuple(args.allowed_tools.split(",")) if args.allowed_tools else DEFAULT_ALLOWED_TOOLS,
                timeout_seconds=args.turn_timeout,
                claude_binary=args.claude_binary,
            ),
            session_id=session_id,
        )
        log.info("Claude agent ready: session_id=%s", session_id)

        # 4. Run the loop.
        runner = HarnessRunner(
            server=server,
            agent=agent,
            http_url=url,
            max_turns=args.max_turns,
            max_consecutive_failures=args.max_consecutive_failures,
        )
        summary = runner.run()
    finally:
        http.stop()

    _print_summary(summary, server)
    return 0 if summary.final_result and summary.final_result.get("status") == "completed" else 1


# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hlbench-agent",
        description=(
            "Drive a Claude Code session through one hlbench-pro run. "
            "Preserves the agent's conversation across iterations via "
            "claude --resume. Writes per-turn logs under "
            "<run_dir>/logs/agent_turns/."
        ),
    )

    # ----- run identity -----
    p.add_argument("--env", default="pendulum", help="registered env id (default: pendulum)")
    p.add_argument(
        "--model-slug", default="claude-code-auto",
        help="run.json:model slug (default: claude-code-auto)",
    )
    p.add_argument(
        "--exp-id", default=None,
        help="distinguish multiple runs of the same (model_slug, env); "
             "auto-generated if omitted",
    )
    p.add_argument(
        "--runs-root", default="./runs",
        help="root for runs/<model>/<env>/<exp-id>/ (default: ./runs)",
    )

    # ----- budget overrides -----
    p.add_argument(
        "--budget", type=int, default=None,
        help="episode_budget override (default: env's default, usually 256)",
    )
    p.add_argument(
        "--max-episodes-per-submit", type=int, default=None,
        help="cap per single /submit call (default: env's default)",
    )

    # ----- harness loop -----
    p.add_argument(
        "--max-turns", type=int, default=12,
        help="hard cap on agent turns; force-finalize on overrun (default: 12)",
    )
    p.add_argument(
        "--max-consecutive-failures", type=int, default=3,
        help="break the loop after N consecutive failed agent turns (default: 3)",
    )

    # ----- claude agent -----
    p.add_argument(
        "--model", default="sonnet",
        help="claude --model (alias 'opus'/'sonnet'/'haiku' or full id; default: sonnet)",
    )
    p.add_argument(
        "--permission-mode", default="bypassPermissions",
        choices=("bypassPermissions", "acceptEdits", "default", "auto", "dontAsk", "plan"),
        help="claude --permission-mode (default: bypassPermissions)",
    )
    p.add_argument(
        "--allowed-tools", default="",
        help=(
            "comma-separated claude --allowedTools (default: "
            f"{','.join(DEFAULT_ALLOWED_TOOLS)})"
        ),
    )
    p.add_argument(
        "--turn-timeout", type=int, default=600,
        help="seconds before a single claude --print invocation times out (default: 600)",
    )
    p.add_argument(
        "--claude-binary", default="claude",
        help="path/name of the claude binary (default: 'claude' on PATH)",
    )
    p.add_argument(
        "--session-id", default=None,
        help="reuse an explicit UUID for the agent session (default: auto-generated)",
    )

    # ----- http server -----
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--port", type=int, default=0,
        help="0 = ephemeral OS-assigned port (default; lets parallel runs coexist)",
    )

    # ----- misc -----
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--no-require-claude", dest="require_claude", action="store_false",
        help="skip the PATH check for 'claude' (only useful if you're using --claude-binary "
             "to a custom binary)",
    )
    p.set_defaults(require_claude=True)

    return p.parse_args(argv)


def _print_summary(summary: RunSummary, server: Server) -> None:
    final = summary.final_result or {}
    print()
    print("=" * 60)
    print("hlbench-agent run complete")
    print("=" * 60)
    print(f"  run_dir:           {server.run_dir}")
    print(f"  session_id:        {summary.session_id}")
    print(f"  turns:             {summary.n_turns}")
    print(f"  termination:       {summary.termination_reason}")
    print(f"  status:            {final.get('status')}")
    if final.get("status") == "completed":
        score = final.get("final_score")
        if score is not None:
            print(f"  final_score:       {score:.2f}")
        held_mean = final.get("held_out_mean_return")
        if held_mean is not None:
            print(f"  held_out_mean:     {held_mean:.2f}")
    elif final.get("error"):
        print(f"  error:             {final['error']}")
    print(f"  run.json:          {final.get('run_json_path')}")
    print()


if __name__ == "__main__":
    sys.exit(main())
