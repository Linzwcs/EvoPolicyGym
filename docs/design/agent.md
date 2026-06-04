# Agent Launch Protocol

> Status: design note. This defines how EvoPolicyGym starts and drives external agent harnesses such as Codex, Claude Code, or a custom controller.

## Goal

EvoPolicyGym evaluates a full interactive agent session, not isolated stateless submissions. The harness wrapper starts once, receives the run context, and keeps the same session alive while the agent reads feedback, edits `policy.py`, and submits repeatedly. A harness turn may finish normally; the driver can still send another prompt such as "continue optimizing" into the same session context.

## Vocabulary

| Name | Meaning |
|---|---|
| `Harness` | Adapter for one concrete agent runner, for example Codex, Claude Code, or an internal script |
| `Session` | One long-lived interactive harness session |
| `Launch` | Startup context: run root, writable system path, feedback path, server endpoint, protocol version |
| `Loop` | Thin driver that sends the initial prompt and continue prompts to the same session |
| `Transcript` | Non-scoring record of harness replies and stop reason |
| `Drive` | Outer host-side runner that serves the API, builds `Launch`, and invokes `Loop` |

## Startup Handshake

The server opens a run directory and API server, then builds `Launch`.
For local runs, `host.Drive` owns that outer lifecycle: it starts
`infra/http/Server`, creates `Launch.from_host(...)` from the bound endpoint,
runs the supplied `Loop`, closes the server, and returns a `Trial` with the
agent transcript. It does not judge submissions directly; all scoring still
flows through `/submit` and the server-side judge.

Harness processes start with `workspace/` as their working root. Therefore the
initial prompt and staged `AGENTS.md` use workspace-relative paths such as
`AGENTS.md`, `system/policy.py`, and `feedback/submit_000/`. `Launch.environ()`
also exposes absolute paths for wrappers that prefer not to rely on cwd, but
those paths are not part of the agent-facing file contract:

| Variable | Meaning |
|---|---|
| `EVOPOLICYGYM_API` | Base server URL |
| `EVOPOLICYGYM_INFO_URL` | `GET /info` |
| `EVOPOLICYGYM_TASK_URL` | `GET /task` |
| `EVOPOLICYGYM_SUBMIT_URL` | `POST /submit` |
| `EVOPOLICYGYM_WORKSPACE` | Workspace directory |
| `EVOPOLICYGYM_SYSTEM` | Writable policy project directory |
| `EVOPOLICYGYM_FEEDBACK` | Read-only `workspace/feedback` artifacts |
| `EVOPOLICYGYM_AGENTS` | Staged `workspace/AGENTS.md` rules file |
| `EVOPOLICYGYM_PROTOCOL` | Protocol version |

The initial prompt tells the harness that its cwd is the benchmark workspace,
to read `AGENTS.md` first, edit only `system/`, read `feedback/`, use `/info`,
`/task`, and `/submit`, improve both policy behavior and code structure, and
continue until `/info.state.is_finalized == true`. The harness must not create
extra environment rollout data outside the server API; all observations,
rewards, episode lengths, returns, and candidate-policy scores derived from
environment execution must come from `/submit` and prior `feedback/`.

## Session Invariant

For one benchmark run:

1. `Harness.start(launch)` is called at most once.
2. The returned `Session` handles all turns for the run.
3. A completed turn is not a completed session; the wrapper must preserve context and accept the next prompt.
4. The wrapper must not restart the agent between submits.
5. `/finalize` remains server-owned; the agent never calls it.
6. LLM latency, tool latency, and harness wall time are outside EvoPolicyGym resource limits.

This is the agent-side counterpart to the OJ server model: the agent has autonomy between submits, while the server only judges submitted policy rollouts.

## Harness Adapters

Concrete wrappers should implement:

```python
class Harness:
    def start(self, launch: Launch) -> Session: ...

class Session:
    @property
    def key(self) -> str: ...
    def step(self, message: str) -> Reply: ...
    def close(self) -> None: ...
```

Codex, Claude Code, and custom harnesses differ in process startup and prompt transport, but they share the same `Launch` contract and session invariant.

The first concrete adapter is `agent.Command`. It starts one persistent process,
writes prompts to stdin as JSON Lines, reads one reply JSON object from stdout
per turn, and records non-scoring transcript rows under `logs/<name>.jsonl` plus
stderr under `logs/<name>.stderr.txt`. Its stdout is a protocol channel; wrappers
should send human logs to stderr or files. Adapter environment variables are
merged with `Launch.environ()`, with launch values taking precedence.

`agent.Codex` is the first product-specific adapter. It uses `codex exec` for
the first turn, reads the JSON event stream to discover the Codex session/thread
id, and then calls `codex exec resume <id>` for later turns. The OS process is
per-turn, but the logical agent context remains one continuous Codex session.
It writes per-turn prompt, command, stream, stderr, and transcript files under
`logs/codex_turns/`.

Codex CLI sandboxing is harness configuration, not EvoPolicyGym scoring policy.
The benchmark surface remains the local HTTP API. For live Codex runs that must
call `127.0.0.1`, set `[agent] bypass = true`; the adapter then passes
`--dangerously-bypass-approvals-and-sandbox` and omits `--sandbox` and approval
overrides. That bypass is only for reaching the local EvoPolicyGym API; it does
not permit internet access or local Gym/MuJoCo/Box2D/highway rollouts outside
`/submit`. Server-side policy rollouts still use the configured EvoPolicyGym
runtime sandbox.

`agent.Claude` wraps Claude Code in print mode. It runs
`claude --print --output-format stream-json --verbose`, reads the stream for a
Claude `session_id`, and uses `--resume <id>` on later turns. If no session id
is exposed, it falls back to `--continue` from the same run directory. It
records the same per-turn files under `logs/claude_turns/`, including command,
prompt, stream, stderr, and transcript text. Claude permission mode, allowed
tools, model, and passthrough arguments are adapter configuration; they are not
benchmark resources.

`agent.Kimi` wraps Kimi Code in non-interactive print mode. It runs
`kimi --output-format stream-json -p <prompt>`, reads the stream for Kimi's
`sessionId`, and uses `-S <id>` on later turns. If no session id is exposed, it
falls back to `-C` from the same workspace. It writes per-turn prompt, command,
stream, stderr, and transcript files under `logs/kimi_turns/`. Kimi model and
passthrough arguments are adapter configuration; they are not benchmark
resources. The default model is `kimi-k2`, matching the v1 Kimi harness. Kimi
Code 0.6 prompt mode rejects `--yolo` and `--auto`, so the adapter does not
pass approval-mode flags in `-p` mode.

Request frame:

```json
{"type": "prompt", "turn": 0, "message": "..."}
```

Reply frame:

```json
{"turn": 0, "text": "optional note", "stop": false, "data": {}}
```

## Stop Conditions

`Loop` stops when the run is finalized, the session requests `stop`, or a
configured turn limit is reached. The turn limit is a harness-safety guard, not
a benchmark resource budget. If omitted from a run spec, it defaults to the
episode budget so a run has at most one continue opportunity per budgeted
rollout episode.

`Reply.stop` is terminal: use it only when the session cannot continue in the same context. Do not set it merely because the agent completed one response or one submit attempt.

## Retry Policy

Agent-side time limits belong to the concrete harness or outer scheduler. EvoPolicyGym does not score LLM latency or impose a benchmark wall-time limit on Codex, Claude Code, Kimi, or custom agents.

EvoPolicyGym does retry harness/service failures at the turn boundary. Configure `[agent] retries = N` and optional `retry_backoff = seconds`. `Loop` retries every `Session.step(...)` exception, plus adapter replies marked by `timed_out = true`, `retryable = true`, or non-zero `exit_code`. Retry attempts do not spend episode budget unless the previous attempt already reached `/submit`; the recovery prompt therefore tells the agent to query `/info` and inspect `feedback/` before continuing.

Retry events are written to `logs/harness.log` as `agent.retry` and `agent.retry.exhausted`. Exhaustion stops the run with `reason = "retry_exhausted"`.

## Code Quality Signal

EvoPolicyGym keeps final score focused on adaptation and generalization. Software
engineering quality is tracked as auxiliary evidence: the filesystem store
computes deterministic static metrics for every submitted `system/` snapshot
and writes them to `checkpoints/submit_NNN/metrics.json`, then summarizes best,
per-submit, and trend views under `run.json:outcome.auxiliary`. These metrics
are host-computed and non-scoring. The agent-facing `AGENTS.md` and continue
prompt still encourage clean structure so agents can demonstrate iterative code
maintenance, not just one-off reward hacking.
