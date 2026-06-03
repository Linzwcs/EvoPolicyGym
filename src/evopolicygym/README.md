# EvoPolicyGym Package Layout

`evopolicygym` is the main package for the protocol described in
`docs/protocol`. The package is organized by stability boundary rather
than by implementation convenience.

## Directories

- `core/`: pure vocabulary and ports. No filesystem, HTTP, subprocess,
  Gym, or JSON dependencies.
- `judge/`: protocol state transitions: open a run, accept a submit,
  spend budget, select with validation, and close with final score.
- `protocol/`: version constants and wire-schema builders for `/info`,
  `/task`, `/submit`, `summary.json`, `run.json`, and the packaged
  agent-facing `AGENTS.md` rules.
- `infra/`: concrete adapters. Filesystem stores live in `infra/fs`,
  sandbox/environment runners in `infra/runtime`, and HTTP in `infra/http`.
- `agent/`: harness launch protocol for long-lived agent sessions.
- `envs/`: environment catalog and built-in task registrations. Each built-in
  environment is a small package with executable world code plus `task.md` for
  the agent-facing task description.
- `check/`: run and environment invariant checkers.
- `metric.py`: deterministic host-side static code metrics for submitted
  policy snapshots.
- `host/`: outer assembly helpers that wire envs, stores, runtimes, pools,
  limits, agent-facing services, HTTP servers, and agent loops for runnable
  hosts.
- `config.py`: typed JSON/TOML run specs for CLI-driven local runs.
- `suite.py`: suite expansion, optional parallel execution, and `suite.json` reporting for batches of
  existing run specs.
- `cli.py`: module entry point for local runs, exposed as the `evopolicygym`
  console script when the package is installed.

## Dependency Rule

Dependencies point inward only:

```text
host -> infra, envs, judge, protocol, agent
infra -> judge, protocol, core, metric
agent -> protocol
suite -> config
envs, check -> core
judge, protocol -> core
metric -> standard library
```

`core` must stay dependency-free except for the Python standard library.
`judge` may depend on `core` and `protocol`, but not on `infra`.
`host` is the outermost composition layer and may depend on concrete adapters.

Storage ports are split by artifact responsibility: `Runs` manages run
lifecycle state, `Snaps` creates immutable checkpoints, `Feeds` records
agent-visible train feedback, `Evals` records hidden validation/final
scores, and `Works` materializes the final best workspace. `Store` is the
full facade for adapters that implement all artifact responsibilities.
`Feeds.feed` persists a core `Report`, which carries the judged `Feed` plus
timing metadata; adapters return the exact agent-visible summary they wrote.
Snapshots carry the original submit cost so hidden validation/final eval can
construct non-leaking `env_meta` without exposing hidden pool size.
`Case`, `Turn`, and `World` are core runtime boundaries: env packages provide
worlds, while runtime rollers map pool-local integer ids into `Case` objects
and turn episodes into traces.

`infra/fs/FileStore` is the first concrete store. It treats its `root` as a
single run directory, creates `workspace/system`, `checkpoints/`,
`workspace/feedback`, `logs/`, and `workspace/AGENTS.md`, writes
`summary.json`, text episode artifacts, submit-level and per-episode error
files, and `run.json`, then mirrors the selected best checkpoint back into
`workspace/system` during close. It also appends framework events to
`logs/harness.log` as JSONL. The staged rules file is hashed into
`run.json:versions.agents_md_hash`. Each submit checkpoint also gets a
host-computed `metrics.json`; final summaries expose `code_metrics_best`,
`code_metrics_by_submit`, and `code_metrics_trend` under
`run.json:outcome.auxiliary` without affecting selection or score.

`infra/runtime/PolicyRuntime` is the first runtime skeleton. It validates
and imports `policy.py`, initializes `Policy` with protocol `env_meta`, and
delegates episode execution to a `Roller` adapter so Gym-specific rollout
logic can be added without changing judge flow.
The default `Roller` consumes a minimal `World` adapter: `reset(case: Case)`,
`step(action) -> Turn`, and `sample()`. It turns policy episodes into core
`Trace` records while keeping Gym wrappers outside the judge/core layers.
`PolicyRuntime` captures successful `Policy.__init__` stdout/stderr into the
first train episode, and `Roller` captures per-episode `reset`/`act` streams.
`SandboxRuntime` is the first subprocess-backed runtime. It executes import,
init, train submit execution, and hidden eval in short-lived child processes,
optionally limits rollout execution with `Sandbox(rollout=...)`, and can be
selected through `host.local(..., sandbox=Sandbox(...))`. Rollout timeout is
disabled by default and applies only after the agent submits code; agent LLM
latency and thinking time are outside this package's control. `Sandbox`
validates rollout, memory, and multiprocessing context settings up front. Child
crashes that produce no result are reported with the worker exit code;
execute-stage crashes map to submit-level `oom`, which is the closest protocol
verdict for server-detected process loss.

`infra/http/Service` is the framework-neutral agent API facade. It exposes
the `/info`, `/task`, and `/submit` shapes as plain dataclasses, parses
`env_instances`, delegates submit judging to `JudgeSubmit`, and, when hidden
validation/final pools are configured, automatically closes the run once the
episode budget is exhausted.
`infra/http/Server` is the first concrete HTTP binding. It uses the Python
standard library to serve `/info`, `/task`, and `/submit`, returns 405 for
agent-owned `/finalize`, and keeps all protocol behavior inside `Service`.

`agent/Launch`, `Harness`, `Session`, and `Loop` define the agent-side startup
protocol. `agent.Command` is the first concrete adapter: it starts one
persistent JSONL stdio process in `workspace/`, keeps the same session alive for
the whole run, logs transcript rows under `logs/`, and leaves scoring to
server-side submits.
`agent.Codex` wraps OpenAI Codex CLI with `codex exec` and `codex exec resume`,
preserving one logical Codex session across EvoPolicyGym turns. `agent.Claude`
wraps Claude Code print mode with stream JSON output and `--resume`, preserving
the same logical session shape without changing judge flow. `agent.Kimi` wraps
Kimi Code with stream JSON output, `-S` resume, and `-C` fallback from the same
workspace root.

`host.local(...)` is the first complete local assembly path. It opens a
filesystem-backed run, derives smoke-test pools from an `Env` or loads formal
pools from an external data directory, wires `FileStore`, `PolicyRuntime`,
`Roller`, and `Service`, and returns a
`Host` object for tests or future server adapters.
`host.Drive` is the first run-level orchestrator. It serves a `Host` through
the stdlib HTTP adapter, builds `Launch` from the bound endpoint, runs a
persistent `Loop`, then returns a `Trial` containing the launch context and
agent transcript. The convenience function `host.drive(host, loop)` exposes the
same path for tests and future CLIs.
`suite.Suite` expands a JSON/TOML experiment matrix into serial `Job` objects
whose specs use the same single-run path. Suite jobs are laid out as
`<suite.root>/<model>/<env>/<exp-id>/`, then the aggregate `suite.json` is
written under the suite root.

The current CLI is intentionally thin. It composes the same library objects and
does not add scoring behavior:

```bash
uv run evopolicygym run \
  --env toy --runs runs --model script --exp-id smoke-001 \
  --budget 8 --maximum 1 \
  --agent command -- python agent.py
```

The same run can be driven from a JSON or TOML spec:

```toml
[run]
env = "toy"
runs = "runs"
model = "codex"
exp_id = "smoke-001"
budget = 8
maximum = 1

[agent]
kind = "codex"
bypass = true  # live Codex may need this to reach the local HTTP API
```

If `[agent] limit` is omitted, it defaults to the run episode budget. Set it
explicitly only when the harness needs a different turn guard.

Use Claude Code by switching the agent section:

```toml
[agent]
kind = "claude"
model = "sonnet"
```

Use Kimi Code similarly:

```toml
[agent]
kind = "kimi"
model = "kimi-k2"
```

Then run:

```bash
uv run evopolicygym run --config run.toml
```

For small batches, use a suite spec:

```toml
[suite]
root = "runs"
repeats = 2
jobs = 2

[[run]]
env = "toy"
budget = 8
maximum = 1

[[agent]]
kind = "claude"
model = "sonnet"
```

```bash
uv run evopolicygym suite --config suite.toml
```

Suite jobs run independently under `<root>/<model>/<env>/<exp-id>/`. The
`jobs` setting controls suite-level parallelism. After each completed job, the
CLI runs the artifact checker and records `category`, `check.ok`, and issue
summaries in `suite.json`.

`Env` is the environment contract: it owns `Task`, judge-only `Secret`, world
factory, optional score `value`, artifact `Caps`, task text loaded from
`task.md`, and pool derivation through `Env.pool(kind)`. `envs/Registry` is the
first environment catalog. `envs/toy/` provides a dependency-free smoke-test
reference, and `envs/cartpole/` provides the first non-toy control benchmark
through the same `World.reset/step/sample` adapter.

`check.check(run_dir)` validates completed run artifacts without mutating
them. The first checker covers `run.json`, submit ordering, summary basics,
budget conservation, staged `AGENTS.md` hash, text episode artifacts,
validation-score coverage, checkpoint code metrics, metrics auxiliary
consistency, and best-checkpoint workspace mirroring.
`check.check_env(env)` validates an in-memory environment registration before it
enters the catalog: task and secret metadata, pool/case mapping, hidden ref
leaks, world smoke behavior, artifact capabilities, and final score value shape.
With `check.check_env(env, root)`, it also validates external case split files
`train.json`, `valid.json`, and `heldout.json` for existence, shape, size,
duplicates, and split overlap.
