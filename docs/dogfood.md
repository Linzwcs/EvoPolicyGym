# Automated Evaluation with Claude Code, OpenAI Codex, or Moonshot Kimi

The `hlbench agent` subcommand drives a complete `init → submit → finalize`
loop end-to-end against any registered hlbench-pro env, using one of
three LLM coding-agent CLIs as the policy author:
[Claude Code](https://claude.com/claude-code),
[OpenAI Codex CLI](https://github.com/openai/codex), or
[Moonshot Kimi Code](https://moonshotai.github.io/kimi-code/).
Pick the backend with `--backend {claude,codex,kimi}` (default
`claude`).

The key property: **the agent's conversation context is preserved
across iterations**. Every turn after the first uses session resume
(`claude --print --resume <uuid>`, `codex exec resume <id>`, or
`kimi -S <session-id> -p`), so the inner agent sees its own prior
reasoning, prior code edits, and prior submit feedback as ordinary
chat history. Contrast with the `../hlbench` reference (which
spawns a fresh subprocess per epoch and relies on workspace files
alone to carry state forward).

## Backends at a glance

| Capability | `--backend claude` | `--backend codex` | `--backend kimi` |
|---|---|---|---|
| Default model | `sonnet` | `gpt-5-codex` | `kimi-k2` |
| Session id | pre-allocated UUID4 (`claude --session-id`) | scraped from turn 0's `session_meta` event | scraped from stream-json (or `~/.kimi-code/session_index.jsonl` keyed by workDir as fallback) |
| Resume command | `claude --print --resume <uuid>` | `codex exec resume <session-id>` | `kimi -S <session-id> -p` |
| Default `--turn-timeout` | 600 s | 900 s | 900 s |
| Per-turn token / cost | surfaced (claude `--output-format=json` carries them) | not surfaced (codex 0.133's `--json` has neither yet) | not surfaced (kimi 0.6's stream-json has neither yet) |
| Sandbox bypass flag | `--claude-permission-mode bypassPermissions` | `--codex-bypass-approvals` (default on) | `-y` yolo (default on; disable with `--no-kimi-yolo`) |
| Session rollout file | inside `<run_dir>/logs/agent_turns/` only | also persisted at `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` | also persisted at `~/.kimi-code/sessions/wd_<basename>_<hash>/session_<uuid>/` (load-bearing for resume) |

All three backends produce the same `<run_dir>/logs/` shape
(`harness.log`, `harness_runner.json`, `agent.jsonl`,
`agent_turns/turn_NNN.{...}`) so analyst tools work unchanged.

## Prerequisites

Common:

- The `hlbench` package installed: `uv pip install --no-deps -e .`
  followed by `uv pip install gymnasium numpy` (per the
  [quickstart](./quickstart.md)).

For `--backend claude`:

- `claude` (Claude Code CLI) on `$PATH`. Verify with `which claude`.
- A logged-in / API-keyed Claude Code (the harness inherits whatever
  auth the local CLI is configured with).

For `--backend codex`:

- `codex` (OpenAI Codex CLI 0.133+) on `$PATH`. Install with
  `npm i -g @openai/codex` or follow the upstream README. Verify
  with `codex --version`.
- Auth via `codex login` (writes `~/.codex/auth`) or set
  `OPENAI_API_KEY`.

For `--backend kimi`:

- `kimi` (Moonshot Kimi Code 0.6+) on `$PATH`. Install via the
  upstream installer at https://moonshotai.github.io/kimi-code/ —
  it typically lands at `~/.kimi-code/bin/kimi` and adds the
  directory to `PATH` via your shell rc. Verify with
  `kimi --version`.
- Auth via `kimi login` (interactive; writes provider/model entries
  into `~/.kimi-code/config.toml`).

## One-shot: drive a Pendulum run

With **Claude Code** (default backend):

```bash
.venv/bin/hlbench agent \
    --env pendulum \
    --budget 32 \
    --max-turns 8 \
    --model sonnet \
    --runs-root ./runs \
    --exp-id dogfood-1
```

With **OpenAI Codex**:

```bash
.venv/bin/hlbench agent \
    --backend codex \
    --env pendulum \
    --budget 32 \
    --max-turns 8 \
    --model gpt-5-codex \
    --model-slug codex-auto \
    --runs-root ./runs \
    --exp-id dogfood-codex-1
```

With **Moonshot Kimi**:

```bash
.venv/bin/hlbench agent \
    --backend kimi \
    --env pendulum \
    --budget 32 \
    --max-turns 8 \
    --model kimi-k2 \
    --model-slug kimi-auto \
    --runs-root ./runs \
    --exp-id dogfood-kimi-1
```

What this does:

1. Initializes a fresh run dir at
   `./runs/<model-slug>/pendulum/<exp-id>/` (model slug is
   overridable with `--model-slug`; default `claude-code-auto`).
2. Starts the hlbench HTTP server on an ephemeral port (default;
   pass `--port 8765` if you want a fixed port).
3. For **claude**: generates a UUID and spawns the first turn with
   `claude --print --session-id <uuid> --output-format stream-json`.
   For **codex**: spawns `codex exec --json -C <workspace> --skip-git-repo-check
   --dangerously-bypass-approvals-and-sandbox -m <model> <prompt>`
   and scrapes the codex session id from turn 0's `session_meta`
   event. Either way, the initial prompt includes workspace path,
   server URL, full task description, `GET /info` JSON, AGENTS.md
   highlights, and operating instructions.
4. After the first turn returns, every subsequent turn resumes:
   `claude --print --resume <uuid>` or `codex exec resume <id>`.
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

Shared (apply to both backends):

| Flag | Default | What it controls |
|---|---|---|
| `--backend` | `claude` | Pick the agent CLI backend: `claude`, `codex`, or `kimi` |
| `--env` | `pendulum` | Which registered env to run |
| `--budget` | env's default (256 for Pendulum) | `episode_budget` override |
| `--max-episodes-per-submit` | env's default | Per-submit cap |
| `--max-turns` | 12 | Hard cap on harness turns; auto-finalize on overrun |
| `--model` | `sonnet` (claude) / `gpt-5-codex` (codex) / `kimi-k2` (kimi) | Model id passed to the chosen backend |
| `--turn-timeout` | 600 s (claude) / 900 s (codex/kimi) | Per-turn subprocess timeout |
| `--max-consecutive-failures` | 3 | Loop bails after N back-to-back failures |
| `--port` | `0` (ephemeral) | HTTP server port |
| `--exp-id` | auto | `runs/<model>/<env>/<exp-id>/` segment |
| `--model-slug` | `claude-code-auto` | `run.json:model` field |
| `--session-id` | auto | Reuse a UUID (claude only honors it as the literal session id; codex uses it as the harness label) |

Claude-only:

| Flag | Default | What it controls |
|---|---|---|
| `--claude-permission-mode` | `bypassPermissions` | claude tool-permission mode (alias `--permission-mode` deprecated) |
| `--claude-allowed-tools` | `Bash,Read,Edit,Write,Glob,Grep` | claude `--allowedTools` (alias `--allowed-tools` deprecated) |
| `--claude-binary` | `claude` | Path/name of the claude binary |
| `--no-require-claude` | (off) | Skip the PATH check for `claude` |

Codex-only:

| Flag | Default | What it controls |
|---|---|---|
| `--codex-binary` | `codex` | Path/name of the codex binary |
| `--codex-sandbox-mode` | `workspace-write` | codex `-s` policy (only used with `--no-codex-bypass-approvals`) |
| `--no-codex-bypass-approvals` | (off, i.e. bypass is on) | Drop `--dangerously-bypass-approvals-and-sandbox`; falls back to `-s <mode>` + interactive approvals |
| `--no-require-codex` | (off) | Skip the PATH check for `codex` |

Kimi-only:

| Flag | Default | What it controls |
|---|---|---|
| `--kimi-binary` | `kimi` | Path/name of the kimi binary (kimi-code typically installs to `~/.kimi-code/bin/kimi`) |
| `--no-kimi-yolo` | (off, i.e. yolo is on) | Drop `-y`; falls back to interactive approvals |
| `--no-require-kimi` | (off) | Skip the PATH check for `kimi` |

## Programmatic use

```python
from pathlib import Path

from hlbench.core.server import Server
from hlbench.http_server import HlbenchHTTPServer
from hlbench_harness import (
    ClaudeAgent, ClaudeAgentConfig,
    CodexAgent, CodexAgentConfig,
    KimiAgent, KimiAgentConfig,
)
from hlbench_harness.runner import HarnessRunner

server = Server(env_id="pendulum", runs_root=Path("./runs"),
                model="my-experiment", exp_id="trial-1",
                config_overrides={"episode_budget": 32})

with HlbenchHTTPServer(server, port=0) as http:
    url = f"http://{http.host}:{http.port}"
    # Pick one:
    agent = ClaudeAgent(
        workspace_dir=server.workspace_dir,
        http_url=url,
        log_dir=server.run_dir / "logs" / "agent_turns",
        config=ClaudeAgentConfig(model="haiku", timeout_seconds=300),
    )
    # ... or:
    # agent = CodexAgent(
    #     workspace_dir=server.workspace_dir,
    #     http_url=url,
    #     log_dir=server.run_dir / "logs" / "agent_turns",
    #     config=CodexAgentConfig(model="gpt-5-codex"),
    # )
    # ... or:
    # agent = KimiAgent(
    #     workspace_dir=server.workspace_dir,
    #     http_url=url,
    #     log_dir=server.run_dir / "logs" / "agent_turns",
    #     config=KimiAgentConfig(model="kimi-k2"),
    # )
    runner = HarnessRunner(server=server, agent=agent,
                           http_url=url, max_turns=8)
    summary = runner.run()
    print(f"final_score = {summary.final_result['final_score']}")
```

Both backends satisfy the same `AgentLike` Protocol (`session_id`,
`turn_count`, `run_turn(prompt) -> TurnResult`), so the runner works
unchanged.

## Testing without a real CLI binary

The unit tests don't depend on `claude`, `codex`, or `kimi` being
installed. Each backend's wrapper takes a `--*-binary` override;
tests pass a tiny Python stub that mimics the JSONL output the real
CLI would produce. See:

- `tests/test_harness_claude_agent.py` — claude stub pattern.
- `tests/test_harness_codex_agent.py` — codex stub pattern (must
  emit a `session_meta` event so the harness can scrape the codex
  session id; a "no-session_meta" stub exercises the fallback path
  where turn 1 starts a fresh `codex exec` rather than `exec resume`).
- `tests/test_harness_kimi_agent.py` — kimi stub pattern (covers the
  three-tier session-id resolution: stream-scrape → session_index.jsonl
  fallback → ``-C`` continue fallback).

The runner itself uses an in-process `FakeAgent`
(`tests/test_harness_runner.py`) that performs scripted Server
actions per turn — useful for verifying loop control (termination,
auto-finalize, failure counter) without an LLM in the loop.

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
