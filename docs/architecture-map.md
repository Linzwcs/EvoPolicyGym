# Code Map (as of 2026-05-30)

Current-state architecture overview. Companion to `architecture.md`
(historical MVP planning) — this doc is what's actually on disk today
and stays in sync with code. Read this first if you're touching the
codebase.

For *protocol* contracts read `SPEC.md` / `AGENTS.md` / `docs/output.md`
/ `docs/submit-protocol.md`. For *philosophy* / invariants read
`CLAUDE.md`.

---

## 1. Layer diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          consumers (outside src/)                    │
│                                                                      │
│  hlbench_cli/        ←─── thin HTTP client (init/serve/info/submit/  │
│                            finalize subcommands)                     │
│                                                                      │
│  hlbench_harness/    ←─── automated eval driver; spawns claude       │
│                            --print and reads back stream events      │
│                                                                      │
│  agents/             ←─── reference policies (e.g. pd_pendulum)      │
│                                                                      │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             │ both: HTTP + lib
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  src/hlbench/  (the server library)                  │
│                                                                      │
│  http_server.py            ┌─────────────────────────────────────┐   │
│  (stdlib http.server       │     core/server.py:Server           │   │
│   thin wrapper around      │     (the per-run entry point)       │   │
│   Server methods)          │                                     │   │
│         ────────────────► │  info()  submit()  finalize()  ...  │   │
│                            └────┬───────┬──────────┬────────────┘   │
│                                 │       │          │                 │
│                            ┌────▼───┐ ┌─▼────────┐ ▼                 │
│                            │ Submit │ │ Heldout  │ Scoring + write   │
│                            │Handler │ │ evaluator│ + auxiliary       │
│                            │ (7    │ │         │  metrics            │
│                            │ phase) │ │         │                    │
│                            └───┬────┘ └────┬────┘                    │
│                                │           │                          │
│                            ┌───▼───────────▼───┐                     │
│                            │     Sandbox       │                     │
│                            │  (spawn child +   │                     │
│                            │   SIGALRM act_to  │                     │
│                            │   + meta_path     │                     │
│                            │   denied_import)  │                     │
│                            └────────┬──────────┘                     │
│                                     │                                 │
│                            ┌────────▼─────────┐                       │
│                            │   run_episode    │                       │
│                            │  (gymnasium env  │                       │
│                            │   loop, pure fn) │                       │
│                            └────────┬─────────┘                       │
│                                     │                                 │
│                            ┌────────▼─────────┐                       │
│                            │  envs/registry   │                       │
│                            │  ┌─────────────┐ │                       │
│                            │  │ pendulum/   │ │                       │
│                            │  └─────────────┘ │                       │
│                            └──────────────────┘                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module catalog

### 2.1 `src/hlbench/` — server library (no consumers below this line)

| Module | LOC | Role |
|---|---:|---|
| `core/server.py` | 605 | Per-run orchestrator. `info()` / `submit()` / `finalize()`. Stages workspace, owns `SubmitState`, writes `run.json`. |
| `core/submit_handler.py` | 804 | 7-phase submit lifecycle (request → snapshot → validate → compile → init → execute → commit). Owns the 11-verdict enum. |
| `core/sandbox.py` | 541 | `multiprocessing(spawn)` child holding one Policy. SIGALRM-based `act()` wall-time guard. `_DeniedImportFinder` on `sys.meta_path`. stdout/stderr capture. |
| `core/env_runner.py` | 259 | Pure function `run_episode(policy, env, ...) -> EpisodeRecord`. Knows nothing about feedback, submits, budget. |
| `core/feedback.py` | 252 | Atomic writers for `summary.json`, `trajectory.jsonl`, `error.txt`, `stdout/stderr.txt`. Filename + width helpers. NaN/Inf JSON encoding. |
| `core/heldout.py` | 127 | One-shot held-out evaluator. Snapshots `workspace/system/`, runs M held-out seeds in a fresh Sandbox. Server.finalize() calls this once. |
| `core/scoring.py` | 151 | Pure functions: `normalized_score`, `final_score` (clip 0..1.2 × 100), `auc_in_loop`, `episodes_to_threshold`, `build_auxiliary`. |
| `core/seed_resolver.py` | 50 | Loads `train.json` + `heldout.json`. Resolves agent-facing integer IDs → hidden real seeds. |
| `core/harness_log.py` | 104 | Plain-text lifecycle log writer for `<run_dir>/logs/harness.log`. No-op via `.disabled()`. |
| `http_server.py` | 201 | stdlib `http.server` thin wrapper. 4 endpoints: `GET /info`, `GET /task`, `POST /submit`, `POST /finalize`. |
| `envs/registry.py` | 128 | `EnvDefinition` dataclass + `register_env()` + `get_env()`. |
| `envs/pendulum/` | 145 | The only registered env. `__init__.py` (registration call), `starter_policy.py` (zero-torque skeleton), `TASK.md`, `train.json`, `heldout.json`. |

### 2.2 `hlbench_cli/` — argparse HTTP client

| Module | LOC | Role |
|---|---:|---|
| `main.py` | 284 | `hlbench {init,serve,info,submit,finalize}`. `init` + `serve` use `Server` directly (lib); the rest are pure HTTP clients. |

### 2.3 `hlbench_harness/` — automated Claude Code driver

| Module | LOC | Role |
|---|---:|---|
| `__main__.py` | 252 | `hlbench-agent` CLI. Spins up Server + HTTP background thread + ClaudeAgent + HarnessRunner. |
| `runner.py` | 353 | `HarnessRunner` loop. Termination priority: `budget_exhausted` → `agent_finalized` → `consecutive_failures` → `max_turns`. Always auto-finalizes. |
| `claude_agent.py` | 409 | `ClaudeAgent` subprocess wrapper around `claude --print --output-format=stream-json`. Pre-allocates UUID; `--session-id` on turn 0, `--resume` on turn 1+. Streams events to `turn_NNN.stream.jsonl` in real time. |
| `prompts.py` | 303 | `compose_initial_prompt()` (full task + /info JSON + AGENTS.md excerpt + ops instructions) and `compose_continuation_prompt()` (terse: turn header + last submit recap + nudge). |
| `state.py` | 102 | `TurnObservation` — bridges live `Server.info()` and on-disk `summary.json` files. |
| `agent_log.py` | 127 | Per-output.md §6.2 JSONL writer for `<run_dir>/logs/agent.jsonl`. `agent_start` / `completion` / `agent_end`. |

### 2.4 `agents/` — reference policies (consumers)

| Path | Purpose |
|---|---|
| `agents/pd_pendulum/policy.py` | Reference energy-shaping swing-up + PD stabilize. ~50 LOC. Drop into `workspace/system/policy.py` to baseline. |

---

## 3. Lifecycle flows

### 3.1 Server lifecycle (used by both CLI and harness)

```
Server(env_id, runs_root, model, exp_id, config_overrides)
  ├── get_env(env_id)              → EnvDefinition
  ├── SeedResolver(train, heldout)  → ID → real_seed resolver
  ├── compute run_dir              = runs_root / model / env_id / exp_id
  ├── mkdir workspace + checkpoints + logs
  ├── copy env.starter_policy_path → workspace/system/policy.py  (if missing)
  ├── copy AGENTS.md               → workspace/AGENTS.md
  ├── SubmitState(remaining_budget=N)
  ├── HarnessLog(run_dir/logs/harness.log)
  └── SubmitHandler(env, sm, ws, cfg, checkpoints_dir, harness_log)

Server.info()    → /info JSON (static config + live state)
Server.submit()  → SubmitHandler.handle(env_instances, state) → SubmitResult
Server.finalize() → snapshot ws/system → evaluate_heldout(snapshot) → scoring →
                    write run.json → cache FinalResult
```

### 3.2 SubmitHandler 7-phase lifecycle (submit_handler.py)

```
handle(env_instances, state):
  Phase 1  Request     → invalid_env_instance | budget_invalid  (no budget consumed)
  ── snapshot taken; budget committed from here on ──
  Phase 2  Snapshot    → missing_policy (system/ missing)
  Phase 3  Validate    → missing_policy (no policy.py)
  Phase 4  Compile     → import_error (sandbox-side)
  Phase 5  Initialize  → init_error | init_timeout | denied_import
  Phase 6  Execute     → loop episodes; submit_wall_exceeded | oom (partial preserved)
  Phase 7  Commit      → write summary.json atomic + checkpoint snapshot
```

Each phase's failure is recorded by one of `_fail_pre_consume` /
`_fail_post_consume` / `_fail_partial_execute`. Per-episode failures
inside Phase 6 (`reset_error` / `act_error` / `act_timeout`) DON'T
change the submit verdict — they go in per-episode `error.txt`.

### 3.3 Harness loop (hlbench_harness/runner.py)

```
HarnessRunner.run():
  agent_log.agent_start(model=..., session_id=...)

  for turn_idx in range(max_turns):
    prompt = compose_initial_prompt(...) if turn_idx == 0
             else compose_continuation_prompt(observe(server, ws))
    result = agent.run_turn(prompt)        # streams to turn_NNN.stream.jsonl
    post_info = server.info()
    record TurnLogEntry; emit agent_log.completion(...)

    if consecutive failures ≥ cap: break  (termination = consecutive_failures)
    if post_info.is_finalized:    break  (termination = agent_finalized)
    if remaining_budget == 0:     break  (termination = budget_exhausted)
  else:
    termination = max_turns

  server.finalize()                        # idempotent — always called
  agent_log.agent_end(reason=termination, ...)
  write run_dir/logs/harness_runner.json
```

### 3.4 ClaudeAgent subprocess (claude_agent.py)

```
ClaudeAgent.run_turn(prompt):
  cmd = ["claude", "--print", "--output-format", "stream-json", "--verbose",
         "--session-id"|"--resume", <uuid>,
         "--permission-mode", "bypassPermissions",
         "--allowedTools", "Bash Read Edit Write Glob Grep",
         "--model", <sonnet|opus|haiku|id>,
         <prompt>]

  Popen(cmd, cwd=workspace, env={HLBENCH_URL, HLBENCH_SESSION_ID, ...})
  ├── reader thread:  for line in proc.stdout:
  │                       write to turn_NNN.stream.jsonl + flush
  │                       parse; if type=="result" capture as last_result
  └── main thread:    proc.wait(timeout)
                      if timeout: kill + drain

  extract cost_usd / num_turns / usage / text from last_result
  write turn_NNN.json (just result event) + turn_NNN.txt (transcript)
  return TurnResult(...)
```

---

## 4. Cross-cutting design rules

### 4.1 Library / consumer separation

`src/hlbench/` contains the server lib only. Anything that USES
`Server` (CLI, agents, harness, scripts) lives outside. The HTTP
wrapper (`http_server.py`) is a carve-out — it's transport for the
lib, not a consumer of it.

### 4.2 HTTP-first for agents

External agents (Claude, codex, custom scripts) MUST use the HTTP
endpoints. The Python lib API is for tests + harness orchestration
only. Per CLAUDE.md invariant 8.

### 4.3 Held-out invisibility

The agent NEVER sees:
- held-out seeds, held-out pool size, individual held-out returns
- `expert_baseline`, `random_baseline`
- the env_instance → real seed mapping

`EnvDefinition.public_env_meta()` strips these before they reach
`/info:env_meta`. SeedResolver keeps the held-out array internal.

### 4.4 Atomic feedback writes

`summary.json` is written via temp-file + rename — agents never see a
half-written file. Per-submit `trajectory.jsonl` / `stdout.txt` /
`stderr.txt` are written before `summary.json` appears, so "if
summary.json is here, the rest is too" holds.

### 4.5 64KB caps on log/error files

`errors.txt` / per-episode `error.txt` cap at 64KB cumulative across
appended events. Beyond cap, a single `category: "truncated"` sentinel
is written, further events dropped. `stdout.txt` / `stderr.txt` cap
at 64KB per episode with `... [truncated at 64KB] ...` marker.

---

## 5. On-disk artifact map

After a complete run with `hlbench-agent`:

```
runs/<model>/<env>/<exp-id>/
├── run.json                         # headline: final_score + outcome
├── workspace/                       # final agent-facing view
│   ├── AGENTS.md
│   ├── system/policy.py             # the agent's final code
│   └── feedback/submit_NNN/         # one per submit
│       ├── summary.json
│       ├── errors.txt               # ◯ submit-level failure only
│       └── episodes/ep_<XXX>/       # one per attempted episode
│           ├── trajectory.jsonl
│           ├── stdout.txt
│           ├── stderr.txt
│           └── error.txt            # ◯ episode-level failure only
├── checkpoints/submit_NNN/          # per-submit code snapshot + _meta.json
└── logs/
    ├── harness.log                  # lifecycle events (server-side)
    ├── harness_runner.json          # per-turn timeline (harness-side)
    ├── agent.jsonl                  # agent_start / completion / agent_end
    └── agent_turns/
        ├── turn_NNN.prompt.txt      # exact prompt sent
        ├── turn_NNN.stream.jsonl    # full streaming events (thinking, tool_use, ...)
        ├── turn_NNN.json            # just the terminal result event
        └── turn_NNN.txt             # human-readable transcript
```

---

## 6. Testing surface

| Test file | What it covers |
|---|---|
| `test_skeleton.py` | Package import + env registration |
| `test_env_runner.py` | `run_episode` with reference PD on Pendulum |
| `test_env_runner_edges.py` | Reset error, act error, act timeout, missing methods |
| `test_sandbox.py` | Subprocess lifecycle, denied imports, init timeout, OOM signal |
| `test_submit_handler.py` | 7-phase lifecycle, every verdict, partial execute |
| `test_server_e2e.py` | Full init → submit → finalize, run.json schema, checkpoints, starter staging |
| `test_http_server.py` | 4 endpoints, error paths, JSON shape |
| `test_scoring.py` | normalized_score, final_score, AUC, episodes_to_Npct |
| `test_harness_log.py` | Lifecycle event format, disabled mode |
| `test_cli.py` | `hlbench` CLI parser + each subcommand |
| `test_harness_prompts.py` | Initial + continuation prompt golden tests |
| `test_harness_state.py` | TurnObservation, summary loading, progress line |
| `test_harness_runner.py` | Loop termination, force-finalize, agent.jsonl, cost aggregation |
| `test_harness_claude_agent.py` | Subprocess wrapping, stream-json capture, timeout preservation |
| `test_harness_agent_log.py` | JSONL roundtrip, None-drop, write-failure swallowing |

**167 tests · mypy strict · ruff clean** as of this writing.
