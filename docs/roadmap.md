# EvoPolicyGym Roadmap

> Status: implementation roadmap, not a normative protocol chapter. Protocol
> requirements stay in `docs/protocol/`.

## Goal

EvoPolicyGym should become a complete benchmark harness for evaluating how
agents improve executable policies from budget-limited feedback. The server
must preserve agent autonomy: agents submit freely through the OJ-style API,
while the judge owns budget accounting, hidden validation selection, and final
held-out scoring.

## Current Baseline

The package lives under `src/evopolicygym/`. The core vocabulary, judge
flows, submit `Report` boundary, protocol schema builders, filesystem store,
text episode artifact writer, packaged `AGENTS.md` staging and hashing, policy
runtime stream capture, toy and CartPole environments, artifact checker,
framework-neutral HTTP facade, stdlib HTTP server adapter, subprocess sandbox
runtime, agent launch/session protocol, generic JSONL command harness adapter,
Codex CLI adapter, Claude Code adapter, Kimi Code adapter, local host assembly,
run-level host/agent driver, host-computed static code metrics, typed run
configuration, suite expansion, parallel suite execution, `suite.json`
reporting, and module CLI entry point are in place. The current
test suite uses `unittest` and covers the main submit, close, schema, store,
runtime, roller, env, checker, HTTP, server, sandbox, host, drive, CLI, suite,
toy e2e, and CartPole/Codex CLI smoke paths.
`docs/real-codex-smoke.md` documents the manual live-Codex CartPole smoke
using `docs/examples/cartpole-codex.toml` with budget 16 and
`[agent] bypass = true`; it remains outside automated tests because it depends
on local authentication and network access.

Run the suite with:

```bash
uv run python -m unittest discover -s tests
```

## Freeze First

These contracts should stabilize before broader environment work:

- Core model names and meanings: `Run`, `Task`, `Pool`, `Budget`, `Submit`,
  `Snap`, `Feed`, `Score`, `Pick`, `Verdict`.
- Judge ownership: `JudgeSubmit` handles request-to-feedback; `JudgeClose`
  handles validation pick and final scoring.
- Budget semantics: Phase 1 rejects do not spend budget; all accepted submits
  spend their requested cost.
- Port boundaries: `Store`, `Runtime`, `World`, and `Catalog` remain the
  extension seams for storage, execution, environment rollout, and registry.
- Case mapping: agents submit integer `env_instances`; `Pool.case(i)` maps
  them to runtime `Case(id, ref, data)` objects before `World.reset(Case)`.
  Formal runs load those cases from an external data directory containing
  `train.json`, `valid.json`, and `heldout.json`.
- Env contract: each `Env` owns `Task`, `Secret`, `make`, optional `value`,
  `Caps`, task text, and canonical pool derivation through `Env.pool(kind)`.
- Env checker: `check_env(env)` validates the registration contract, hidden
  metadata leakage, `World.reset(Case)` smoke behavior, artifact caps, and final
  score value shape before an env enters the registry. `check_env(env, root)`
  additionally validates external split files for existence, size, duplicates,
  and split overlap.
- Submit reporting: `JudgeSubmit` creates one `Report`; `Store.feed` writes it
  and returns the exact summary used by HTTP responses.
- Text feedback artifacts: `FileStore` writes `trajectory.jsonl`, `stdout.txt`,
  `stderr.txt`, submit-level `errors.txt`, and per-episode `error.txt` from
  `Report.traces`; `check` validates the basic episode layout and trajectory
  lengths.
- Framework logging: local runs append JSONL framework events to
  `logs/harness.log`, including run open/close, server/loop lifecycle, submit
  phases, feedback writes, hidden eval records, and workspace mirroring.
- Code metrics: every submit snapshot gets host-computed `metrics.json`, and
  `run.json:outcome.auxiliary` exposes best, by-submit, and trend summaries
  for post-hoc software-engineering analysis without affecting score.
- Agent rules file: EvoPolicyGym owns the source `AGENTS.md`, stages it into
  every run at `workspace/AGENTS.md`, exposes its path to harnesses, and records
  `versions.agents_md_hash` in `run.json`.
- Runtime streams: `PolicyRuntime` captures successful `Policy.__init__`
  stdout/stderr into the first train episode, while `Roller` captures policy
  `reset` and `act` stdout/stderr per episode.
- Sandbox runtime: `SandboxRuntime` runs policy import/init/execute/eval in
  short-lived child processes, validates sandbox limits, reports child crashes
  with exit codes, supports optional rollout timeout mapping, and can be
  enabled through `host.local(..., sandbox=Sandbox(...))`.
- Host assembly: `host.local(...)` opens a filesystem-backed run and wires
  `Env`, optional external data splits, `FileStore`, `PolicyRuntime`, `Roller`, `Service`, train
  pool, validation pool, final pool, and submit limits.
- Run driver: `host.Drive` owns the outer lifecycle for local benchmark runs:
  start `infra/http/Server`, build `Launch`, run one persistent `Loop`, close
  the server, and return a non-scoring `Trial` transcript.
- CLI entry: `evopolicygym run` composes registry lookup,
  `host.local`, `Command`, `Loop`, and `Drive` for one local run. It prints a
  small JSON summary and exits nonzero if the run does not finalize.
- Run config: `config.Spec` loads JSON or TOML specs with `[run]`, `[agent]`,
  and `[server]` sections. CLI flags remain available as overrides.
- Server adapter: `infra/http/Server` binds `Service` to real stdlib HTTP
  routes for `/info`, `/task`, and `/submit`, while rejecting agent-owned
  `/finalize`.
- Agent API shape: `/info`, `/task`, `/submit`; no agent-owned finalize.
- Agent launch protocol: `Launch` exposes paths and API URLs, harness adapters
  start exactly one long-lived `Session`, and `Loop` reuses that session across
  all submit/feedback turns until the run finalizes or a terminal session stop
  occurs. A completed harness turn only triggers another continue prompt in the
  same context. `Command` provides a generic JSONL stdio process adapter and
  writes non-scoring transcript/stderr logs under `logs/`.
- Codex adapter: `Codex` uses `codex exec` for the first turn and `codex exec
  resume` after it discovers the Codex session id from the JSON event stream.
  Per-turn process logs live under `logs/codex_turns/`. Live Codex runs can set
  `bypass = true` so the outer harness can call the local HTTP API while
  EvoPolicyGym still controls policy rollout sandboxing.
- Claude adapter: `Claude` uses Claude Code print mode with stream JSON output,
  resumes later turns with the discovered session id, and writes per-turn logs
  under `logs/claude_turns/`.
- Kimi adapter: `Kimi` uses Kimi Code stream JSON output, resumes later turns
  with the discovered `sessionId`, falls back to workspace-local continue, and
  writes per-turn logs under `logs/kimi_turns/`.
- Suite runner: `Suite` expands JSON/TOML `[[run]]` by `[[agent]]` by repeat
  into jobs. `[suite] jobs = N` controls parallel execution. Each job reuses
  the normal single-run path and writes its own `run.json`; the suite root
  records aggregate status, checker status, and failure categories in
  `suite.json`.

## Next Milestones

1. **Environment expansion P0/P1**: implement the generic Gymnasium adapter,
   then add the first lightweight envs (`Pendulum`, `Taxi`, and remaining
   Classic Control / Toy Text). See `docs/envs/roadmap.md`.
2. **Binary artifact writer**: add optional `video.mp4` and
   `observations.npy/.npz` writing behind explicit runtime/world capabilities.
3. **Sandbox hardening phase 2**: add stronger CPU/RSS monitoring, structured
   child-crash categories, and platform-specific documentation for memory
   limits and multiprocessing start methods.
4. **Checker expansion**: validate file-backed case sources, error-file
   invariants, observation/video alignment, and deeper summary/run
   consistency.

## Extension Points

Environment work should be additive. A new environment should register an
`Env` containing `Task`, `Secret`, `make()`, optional `value`, `Caps`, and
agent-facing task text loaded from an environment-local `task.md`, then provide
a `World` adapter with `reset(case: Case)`,
`step(action) -> Turn`, and `sample()`. It should not require changes to
`core`, `judge`, or protocol schema builders.

Runtime work can vary independently: in-process smoke tests, subprocess
sandboxing, container isolation, or remote workers should all satisfy the same
`Runtime` port. Scoring extensions should attach to `Score.value` and
`run.json:outcome.auxiliary` without exposing validation or held-out details to
the agent.

## Release Gate

Before `protocol/v2.0-rc.1`, the project should have a real server entry point,
complete feedback artifacts, expanded checker coverage, and e2e tests that open
a run, submit multiple policies on a non-toy environment, auto-close on budget
exhaustion, and verify `run.json` plus workspace/checkpoint mirroring.
