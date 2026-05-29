# Automated Evaluation with Claude Code

The `hlbench-agent` CLI drives a complete `init → submit → finalize`
loop end-to-end against any registered hlbench-pro env, using
[Claude Code](https://claude.com/claude-code) as the policy author.

The key property: **the agent's conversation context is preserved
across iterations**. Every call to `claude --print` after the first
uses `--resume <session_id>`, so the inner Claude session sees its own
prior reasoning, prior code edits, and prior submit feedback as
ordinary chat history. Contrast with the `../hlbench` reference
(which spawns a fresh `claude` subprocess per epoch and relies on
workspace files alone to carry state forward).

## Prerequisites

- `claude` (Claude Code CLI) on `$PATH`. Verify with `which claude`.
- A logged-in / API-keyed Claude Code (the harness inherits whatever
  auth the local CLI is configured with).
- The `hlbench` package installed: `uv pip install --no-deps -e .`
  followed by `uv pip install gymnasium numpy` (per the
  [quickstart](./quickstart.md)).

## One-shot: drive a Pendulum run

```bash
.venv/bin/hlbench-agent \
    --env pendulum \
    --budget 32 \
    --max-turns 8 \
    --model sonnet \
    --runs-root ./runs \
    --exp-id dogfood-1
```

What this does:

1. Initializes a fresh run dir at
   `./runs/claude-code-auto/pendulum/dogfood-1/` (model slug is
   overridable with `--model-slug`).
2. Starts the hlbench HTTP server on an ephemeral port (default;
   pass `--port 8765` if you want a fixed port).
3. Generates a UUID for the Claude session and spawns the first turn
   with `claude --print --session-id <uuid> --output-format json`,
   passing the initial prompt (workspace path, server URL, full task
   description, `GET /info` JSON, AGENTS.md highlights, operating
   instructions).
4. After the first turn returns, every subsequent turn uses
   `claude --print --resume <uuid>` so the conversation continues.
   The continuation prompt is short — just the delta state
   (remaining budget, last submit's verdict + mean) and a nudge to
   keep iterating.
5. Stops when **any** of (in priority order):
   - `remaining_budget` hits 0 after the most recent turn returns
     (`termination_reason: budget_exhausted` — the preferred path),
   - the agent's subprocess fails or times out N turns in a row
     (`termination_reason: consecutive_failures`, default cap 3),
   - `--max-turns` is reached without budget being spent
     (`termination_reason: max_turns` — safety net),
   - the agent calls `POST /finalize` itself despite the prompt
     telling it not to (`termination_reason: agent_finalized` —
     defensive; rare in practice).

   In every case the harness then calls `Server.finalize()` itself
   (which runs the held-out evaluation and writes `run.json`).
   **The agent never has to call `POST /finalize`** — finalization
   is the harness's job.
6. Prints the headline (`final_score`, `held_out_mean_return`) and
   exits non-zero on errored runs.

## Inspect the run afterwards

```
runs/claude-code-auto/pendulum/dogfood-1/
├── run.json                            ← headline (held_out_mean, final_score)
├── workspace/
│   ├── AGENTS.md
│   ├── system/policy.py                ← the agent's final code
│   └── feedback/submit_*/
│       ├── summary.json
│       └── episodes/ep_*/...
├── checkpoints/submit_*/                ← every snapshot the agent shipped
└── logs/
    ├── harness.log                     ← lifecycle events (output.md §6.1)
    ├── harness_runner.json             ← per-turn timeline + final result
    ├── agent.jsonl                     ← agent activity (output.md §6.2):
    │                                     agent_start / completion / agent_end
    └── agent_turns/
        ├── turn_000.prompt.txt         ← exact prompt sent on turn N
        ├── turn_000.stream.jsonl       ← FULL streaming events (thinking,
        │                                 tool_use, tool_result, result) —
        │                                 captured live so partial runs are
        │                                 recoverable on timeout
        ├── turn_000.json               ← just the terminal `result` event
        │                                 (quick-access mirror)
        └── turn_000.txt                ← human-readable transcript
```

Two files give you the whole picture quickly:

- `logs/harness_runner.json` — the harness's view: how many turns,
  the `termination_reason` (one of `budget_exhausted` /
  `consecutive_failures` / `max_turns` / `agent_finalized`),
  per-turn server-state snapshots, the final `run.json` payload
  mirrored.
- `logs/agent.jsonl` — the agent's view (per `output.md §6.2`):
  one line per event. `agent_start` carries model + session_id;
  one `completion` per turn with token + cost + latency; `agent_end`
  with the termination reason and totals. Append-only, line-oriented
  — easy to `jq`/`grep`.
- `logs/agent_turns/turn_000.txt` — full transcript of turn 0 (the
  initial prompt + Claude's response). `turn_001.txt` etc. for later
  turns.

## Knobs

| Flag | Default | What it controls |
|---|---|---|
| `--env` | `pendulum` | Which registered env to run |
| `--budget` | env's default (256 for Pendulum) | `episode_budget` override |
| `--max-episodes-per-submit` | env's default | Per-submit cap |
| `--max-turns` | 12 | Hard cap on harness turns; auto-finalize on overrun |
| `--model` | `sonnet` | `claude --model` (alias `opus`/`sonnet`/`haiku` or full id) |
| `--turn-timeout` | 600 s | Per-turn subprocess timeout |
| `--permission-mode` | `bypassPermissions` | claude tool-permission mode |
| `--allowed-tools` | `Bash,Read,Edit,Write,Glob,Grep` | claude `--allowedTools` |
| `--max-consecutive-failures` | 3 | Loop bails after N back-to-back failures |
| `--port` | `0` (ephemeral) | HTTP server port |
| `--exp-id` | auto | `runs/<model>/<env>/<exp-id>/` segment |
| `--model-slug` | `claude-code-auto` | `run.json:model` field |
| `--session-id` | auto | Reuse a UUID (e.g., to resume an aborted run later) |

## Programmatic use

```python
from pathlib import Path

from hlbench.core.server import Server
from hlbench.http_server import HlbenchHTTPServer
from hlbench_harness.claude_agent import ClaudeAgent, ClaudeAgentConfig
from hlbench_harness.runner import HarnessRunner

server = Server(env_id="pendulum", runs_root=Path("./runs"),
                model="my-experiment", exp_id="trial-1",
                config_overrides={"episode_budget": 32})

with HlbenchHTTPServer(server, port=0) as http:
    url = f"http://{http.host}:{http.port}"
    agent = ClaudeAgent(
        workspace_dir=server.workspace_dir,
        http_url=url,
        log_dir=server.run_dir / "logs" / "agent_turns",
        config=ClaudeAgentConfig(model="haiku", timeout_seconds=300),
    )
    runner = HarnessRunner(server=server, agent=agent,
                           http_url=url, max_turns=8)
    summary = runner.run()
    print(f"final_score = {summary.final_result['final_score']}")
```

## Testing without `claude`

The unit tests don't depend on a real `claude` binary. The harness's
subprocess wrapper takes a `--claude-binary` override; tests pass a
tiny Python stub that mimics `--print --output-format=json`. See
`tests/test_harness_claude_agent.py` for the pattern. The runner
itself uses an in-process `FakeAgent` (`tests/test_harness_runner.py`)
that performs scripted Server actions per turn — useful for
verifying loop control (termination, auto-finalize, failure
counter) without an LLM in the loop.

## Why one continuous session?

Per-epoch fresh sessions (the `../hlbench` design) push all state
through the workspace: the agent must re-read `feedback/`,
`AGENTS.md`, prior submits, etc. on every turn. That's a lot of
context to reload, and the model can't easily remember its own
*reasoning* — only what it wrote to a file.

Single-session iteration via `--resume`:

- The agent's chain-of-thought from earlier turns persists.
- Continuation prompts can be short (just a state delta) because
  the model already knows the rules and what it tried.
- The harness's job shrinks to "decide when to stop and what state
  to surface in the next prompt."

The trade-off is that the inner Claude conversation grows. With
default budget=32 and ~6 submits, ~12 turns, contexts stay well
within Sonnet's window. For larger runs (budget=256, 30+ turns),
consider `--model opus` for the bigger context, or chunk the run
into multiple sessions and resume across boundaries (not yet
supported by the CLI).
