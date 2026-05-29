# docs/output.md — Run Output Layout

This document defines the on-disk layout and file schemas produced by
a single hlbench-pro evaluation run. It is the contract between the
harness implementation and any downstream consumer (analysis scripts,
dashboards, paper plots, replay tools).

`AGENTS.md` and `SPEC.md` describe the protocol from the agent's
perspective. This document describes the artifacts that result.

---

## 1. Top-Level Layout

```
runs/
└── <model>/
    └── <env>/
        └── <exp-id>/
            ├── run.json                  # top-level run metadata + final score
            ├── workspace/                # final workspace state at run end
            │   ├── AGENTS.md
            │   ├── system/               # final code (== latest successful submit)
            │   └── feedback/             # all submit_NNN/
            ├── checkpoints/              # code snapshots, one per submit
            │   ├── submit_000/
            │   ├── submit_001/
            │   └── ...
            └── logs/
                ├── harness.log           # harness execution log
                ├── agent.jsonl           # agent harness events (tool calls, completions)
                └── env.log               # environment warnings/errors (often empty)
```

Three coordinates fully address one run:

- **`<model>`** — the agent that produced the run (LLM family, baseline name).
- **`<env>`** — the environment identifier (matches `env` in the per-run server's `GET /info`).
- **`<exp-id>`** — distinguishes multiple runs of the same (model, env).

`runs/` is the only directory the harness writes outside the per-run
workspace. Implementations MAY make `runs/` configurable via
`HLBENCH_RUNS_DIR` or a CLI flag, but the structure beneath it is
normative.

---

## 2. ID Naming Conventions

### 2.1 `model`

Slug for the agent identity. Format: `[a-z0-9][a-z0-9-]*`, max 64
characters.

| Examples | Notes |
|---|---|
| `claude-opus-4-7` | Frontier LLM |
| `claude-sonnet-4-6` | |
| `gpt-5-4` | |
| `gemini-2-pro` | |
| `random` | Random-action baseline |
| `expert` | Hand-crafted expert reference |
| `ppo-1m` | RL baseline (1M env steps) |
| `dreamer-v3` | Sample-efficient RL baseline |

For LLM agents, include the model version. For algorithmic baselines,
include enough to disambiguate the configuration (e.g., `ppo-1m` vs
`ppo-10m`).

### 2.2 `env`

Matches the `env` field in the environment registration (and in
`GET /info`). Format: `[a-z0-9][a-z0-9_]*`, max 64 characters.
Examples: `pendulum`, `halfcheetah`, `breakout`, `montezuma_revenge`.

### 2.3 `exp-id`

Distinguishes multiple runs sharing the same (model, env). Two modes:

**User-specified.** Provided via `--exp-id <name>` to `hlbench run`.
Format: `[a-zA-Z0-9][a-zA-Z0-9_.-]*`, max 64 characters. The harness
checks for collision with existing directories under
`runs/<model>/<env>/` and refuses to overwrite without `--force`.

**Auto-generated default** (used when `--exp-id` is omitted):

```
<YYYY-MM-DDTHH-MM-SS>__<short-hash>
```

- Timestamp: ISO-8601 in UTC, `:` replaced with `-`.
- Short hash: first 6 hex chars of `SHA-256(model || env || effective_config_json
  || timestamp_iso8601)`, where `effective_config_json` is the merged
  config the server initialized with. Acts as a tiebreaker when two
  runs start in the same second.

Example: `2026-05-28T10-00-00__a1b2c3`.

The triple `(model, env, exp-id)` is the unique identity of a run.
The directory path `runs/<model>/<env>/<exp-id>/` serves as the
canonical identifier; no separate composite ID is generated.

---

## 3. `run.json` — Top-Level Run Metadata

The single most important file in a run directory. Reading just
`run.json` MUST be sufficient to:

1. Identify the run (model, env, exp-id, configuration).
2. Locate every other artifact (paths to workspace, checkpoints, logs).
3. Extract the headline score and auxiliary metrics for analysis.

`run.json` is written **once at run end** (or at run abort). It is
not partially populated mid-run. For mid-run state inspection, see
`workspace/feedback/submit_NNN/summary.json`.

### 3.1 Schema

```json
{
  "schema_version": "0.1",
  "model": "claude-opus-4-7",
  "env": "halfcheetah",
  "exp_id": "2026-05-28T10-00-00__a1b2c3",

  "experiment_dimensions": {
    "episode_budget": 256,
    "min_episodes_per_submit": 1,
    "max_episodes_per_submit": 200,
    "seed_pool_id": "default",
    "agent_harness": "hlbench@0.1.0a1",
    "model_config": {
      "temperature": 1.0,
      "max_tokens": 8192
    }
  },

  "timing": {
    "start_time": "2026-05-28T10:00:00Z",
    "end_time": "2026-05-28T10:42:13Z",
    "wall_time_seconds": 2533.4
  },

  "outcome": {
    "status": "completed",
    "error": null,
    "final_submit_index": 21,
    "final_score": 73.4,
    "held_out_mean_return": 8542.1,
    "held_out_std_return": 412.6,
    "held_out_returns": [8123.4, 8542.6, 8901.2, 7988.5, 8412.0, "...100 floats total..."],
    "auxiliary": {
      "auc_in_loop": 41.2,
      "episodes_to_50pct": 64,
      "episodes_to_80pct": 138,
      "held_out_gap": 4.7,
      "n_submits": 23,
      "n_successful_submits": 22,
      "episodes_used": 200,
      "mean_episodes_per_submit": 8.7
    }
  },

  "artifacts": {
    "workspace": "workspace/",
    "checkpoints": "checkpoints/",
    "logs_harness": "logs/harness.log",
    "logs_agent": "logs/agent.jsonl",
    "logs_env": "logs/env.log"
  },

  "versions": {
    "harness": "0.1.0",
    "env": "0.1",
    "agents_md_hash": "sha256:8f3a..."
  }
}
```

### 3.2 Field Notes

- **`schema_version`** — bumped on breaking changes to this document.
  Consumers SHOULD validate it before parsing. All schema files in
  this layout (`run.json`, `summary.json`, `_meta.json`)
  carry their own `schema_version`.
- **`outcome.status`** ∈ `{"completed", "aborted", "error"}`.
  - `completed`: submit budget exhausted normally and held-out eval ran.
  - `aborted`: agent or operator stopped the run before budget exhaustion.
    `final_submit_index` may be set to last successful submit, or `null`
    if no submits succeeded.
  - `error`: harness failure (sandbox crash, disk full, etc.).
- **`outcome.error`** — `null` when `status` is `completed` or
  `aborted`. When `status == "error"`, an object:
  ```json
  {
    "type": "SandboxCrash",
    "message": "subprocess killed by SIGSEGV",
    "occurred_at_submit": 17,
    "traceback": "Traceback (most recent call last):\n  ..."
  }
  ```
- **`outcome.final_score`** — the headline number per `SPEC.md §5.2`.
  May be `null` if `status != "completed"`.
- **`outcome.held_out_returns`** — the full 100-element array of raw
  held-out episode returns. Present when `status == "completed"`,
  `null` otherwise. Useful for variance/outlier analysis.
- **`artifacts.*`** — paths are relative to the run directory.
  Consumers that move runs MUST keep paths relative.
- **`versions.agents_md_hash`** — SHA-256 of the AGENTS.md in effect at
  run start (after task overrides applied).

### 3.3 What Must Match What

These cross-file invariants MUST hold:

| `run.json` field | Must match |
|---|---|
| `experiment_dimensions.episode_budget` | server's effective `episode_budget` |
| `outcome.auxiliary.episodes_used` | `sum(summary.json:n_episodes for each successful submit)` |
| `outcome.auxiliary.n_submits` | count of `feedback/submit_*/` directories |
| `versions.env` | server's effective `env_version` |
| `env` | server's effective `env` |

The harness MUST verify these on write. Mismatches indicate a harness
bug and SHOULD raise.

---

## 4. `workspace/` — Final Workspace State

The contents of the agent's workspace at run end. Layout matches
`SPEC.md §1`:

```
workspace/
├── AGENTS.md       (the global AGENTS.md from this run's harness version)
├── system/         (final code: identical to latest successful submit's snapshot)
└── feedback/
    ├── submit_000/
    ├── submit_001/
    └── ...
```

Note: there is no `_run.json` or `_final.json` inside the workspace.
Effective config is served by the server's `GET /info`. Final
aggregated results (held-out score, auxiliary metrics) live in
`run.json` at the run directory root (see §3). The agent has no
access to those values since held-out evaluation happens after
the agent exits.

### 4.1 Invariant

`workspace/system/` at run end MUST be byte-identical to
`checkpoints/submit_<final_submit_index>/`. The harness writes this
last as part of run finalization.

### 4.2 Why Keep It

Although `checkpoints/` already preserves every submit, keeping a
top-level `workspace/` enables quick inspection ("what's the final
code?") without traversing into checkpoints, and provides a known
location for tools that operate on the final policy.

---

## 5. `checkpoints/` — Per-Submit Code Snapshots

```
checkpoints/
├── submit_000/
│   ├── _meta.json
│   ├── policy.py
│   ├── utils.py             (if present at submit time)
│   ├── memory.json          (if present at submit time)
│   └── ...
├── submit_001/
│   └── ...
└── ...
```

One subdirectory per submit. The contents of `submit_NNN/` (excluding
`_meta.json`) are exactly what `workspace/system/` contained at the
moment submit N was issued. This enables:

- Re-running any historical policy on fresh seeds.
- Auditing "what code produced submit N's score".
- Post-hoc analysis (AST diffs, method profiling, etc.) using
  whatever tools the analyst chooses.

The same numbering convention is used inside `workspace/feedback/`.

### 5.1 `submit_NNN/` Numbering Width

`submit_NNN/` directories use **zero-padded width that fits the maximum
possible submit count for the run**, with a floor of 3 digits:

```
width = max(3, len(str(episode_budget)))
```

Examples:

| `episode_budget` | Width | Example names |
|---|---|---|
| ≤ 999 (typical) | 3 | `submit_000`, `submit_023`, `submit_199` |
| 1,000 – 9,999 | 4 | `submit_0000`, `submit_0234`, `submit_9999` |
| 10,000+ | 5+ | scales as needed |

The rationale: every submit consumes at least 1 episode from the
budget, so the maximum submit count equals `episode_budget`.
Fixed-width zero padding preserves lexicographic ordering for tools
that sort by filename. Width is decided once at run start (from the
server's effective `episode_budget`) and never changes during a run.

### 5.2 `submit_NNN/_meta.json`

```json
{
  "schema_version": "0.1",
  "submit_index": 0,
  "submit_time": "2026-05-28T10:01:23Z",
  "n_episodes_requested": 8,
  "remaining_budget_before": 200,
  "remaining_budget_after": 192,
  "snapshot_size_bytes": 2048,
  "snapshot_files": ["policy.py"],
  "import_scan": ["numpy", "math"],
  "validation_status": "ok",
  "validation_errors": []
}
```

`validation_status` uses the same unified enum as
`summary.json:status` (see `SPEC.md §4.1`):

| Value | Meaning |
|---|---|
| `ok` | Snapshot passed all checks, episodes ran |
| `budget_invalid` | Rejected: requested episodes outside `[min, max, remaining]` |
| `missing_policy` | Rejected: no `policy.py` or no `Policy` class |
| `denied_import` | Rejected: snapshot imported a forbidden module |
| `import_error` | Rejected: snapshot import raised (syntax, missing module, etc.) |
| `init_timeout` | Rejected: `Policy.__init__` exceeded `policy_load_wall_s` |
| `init_error` | Rejected: `Policy.__init__` raised |

If validation failed, the snapshot is still preserved (the agent
needs to know what it submitted). The corresponding
`feedback/submit_NNN/summary.json` will have a matching non-`ok`
status.

### 5.3 Storage

Snapshots are stored as plain files (not tarballs or git refs) for
direct inspection. Implementations MAY deduplicate identical files
across submits using hardlinks or content-addressed storage, as long
as the directory contents appear identical to a reader.

---

## 6. `logs/`

The `logs/` directory is **implementer- and analyst-facing only**.
The agent has no read access to any file under `logs/`. Diagnostic
information the agent needs to iterate on its policy MUST be
delivered through `workspace/feedback/` (specifically `summary.json`,
`errors.txt`, `stdout.txt`, `stderr.txt`, and the per-episode JSONL
files) — never through `logs/`.

Implementers MUST NOT route information intended for the agent into
`logs/` as a shortcut. See §9 invariant 9.

### 6.1 `harness.log`

Plain-text log from the harness. Format: one line per event.

```
2026-05-28T10:00:00.123Z INFO  run_start model=claude-opus-4-7 env=halfcheetah exp_id=2026-05-28T10-00-00__a1b2c3
2026-05-28T10:00:01.456Z INFO  submit_received submit_index=0 n_episodes=8
2026-05-28T10:00:01.789Z INFO  snapshot_taken size_bytes=2048
2026-05-28T10:00:02.012Z INFO  episode_start submit=0 episode=0 seed=<hidden>
2026-05-28T10:00:08.345Z INFO  episode_end submit=0 episode=0 return=1234.5
...
2026-05-28T10:42:13.000Z INFO  run_end status=completed final_score=73.4
```

Includes all sandbox events (import denials, timeouts, OOMs) and
harness lifecycle. Does NOT include held-out seed values.

### 6.2 `agent.jsonl`

JSONL of agent harness activity. One JSON object per line.

```jsonl
{"t":"2026-05-28T10:00:00.500Z","event":"agent_start","model":"claude-opus-4-7"}
{"t":"2026-05-28T10:00:05.123Z","event":"completion","input_tokens":12500,"output_tokens":847,"latency_ms":3200,"cost_usd":0.184}
{"t":"2026-05-28T10:00:05.140Z","event":"tool_call","tool":"http_get","args":{"path":"/task"}}
{"t":"2026-05-28T10:00:05.150Z","event":"tool_call","tool":"write","args":{"path":"system/policy.py","bytes":1024}}
{"t":"2026-05-28T10:00:10.200Z","event":"submit","n_episodes":8}
{"t":"2026-05-28T10:01:30.000Z","event":"agent_end","reason":"budget_exhausted"}
```

Useful for: debugging failed runs, computing token/cost metrics,
correlating agent reasoning with submit outcomes.

If the agent harness produces verbose internal traces (e.g., chain-of-
thought completions), implementations MAY truncate these to keep file
size bounded; if so, an explicit `"truncated":true` field MUST be set
on truncated entries.

### 6.3 `env.log`

Plain-text log of warnings or non-fatal errors from the environment
(e.g., MuJoCo physics warnings, deprecation notices). Often empty.

### 6.4 Compression

Logs MAY be compressed (`harness.log.gz`, `agent.jsonl.gz`) to save
space. Consumers MUST handle both compressed and uncompressed
extensions transparently.

---

## 7. Multi-Variable Experiments

When sweeping over experiment dimensions (different budgets, seeds,
agent configurations), encode the variation in `exp-id` rather than
adding hierarchy levels:

```
runs/claude-opus-4-7/halfcheetah/
├── b100__s42__2026-05-28T10-00-00__a1b2c3/
├── b200__s42__2026-05-28T10-15-00__d4e5f6/
├── b500__s42__2026-05-28T10-30-00__789abc/
└── b200__s43__2026-05-28T10-45-00__def012/
```

Convention for sweep prefixes (recommended, not enforced):

- `b<N>` — total episode budget
- `s<N>` — seed pool offset
- `t<N>` — temperature × 10 (e.g., `t10` = 1.0)

The `experiment_dimensions` object in `run.json` is the authoritative
record of all varied parameters. The `exp-id` prefix is a
human-readable convenience.

---

## 8. Reproducibility

### 8.1 Re-running a Single Submit

Given a `checkpoints/submit_NNN/` directory and a matching harness
version, the policy can be re-executed on fresh seeds:

```bash
hlbench replay --checkpoint runs/<...>/checkpoints/submit_005 --episodes 100
```

The harness uses the task definition and seed generation logic from
the recorded versions in `run.json`.

Note: re-running does not exactly reproduce the original episode
returns unless the same in-loop seeds are used. Held-out seeds are
deterministic from the harness version and env version.

### 8.2 Reproducing the Full Run

Bit-exact reproduction requires:

- Same harness version (`run.json:versions.harness`).
- Same env version (`run.json:versions.env`).
- Same AGENTS.md (verified via `versions.agents_md_hash`).
- Same agent harness and model.
- Same `experiment_dimensions`.

Note: LLM-driven runs are **not deterministic** even with identical
inputs (sampling temperature, server-side variation). The benchmark
acknowledges this by reporting variance over multiple runs (different
`exp-id`s) rather than claiming bit-exactness.

### 8.3 What's Hidden

The held-out seed pool is NEVER written to any artifact reachable by
the agent or by post-hoc analysis tools that might leak into a future
agent's training data. The pool is reproducible only from
`(harness_version, env_version)` via a deterministic function inside
the harness.

---

## 9. Implementation Invariants

The harness MUST guarantee:

1. **Atomic run.json write.** `run.json` is written via temp-file +
   rename; partial writes are never observable.
2. **No write to checkpoints after submit.** Once
   `checkpoints/submit_NNN/` exists, its contents are immutable.
3. **Workspace mirror.** At run end, `workspace/system/` is
   byte-identical to the snapshot referenced by
   `outcome.final_submit_index`.
4. **Append-only logs.** `logs/*.log` and `logs/*.jsonl` are written
   in append mode; the harness does not rewrite history.
5. **No held-out leakage.** Held-out seeds and per-episode held-out
   returns NEVER appear in any file under the run directory except as
   already-aggregated statistics in `run.json`.
6. **Path stability.** All paths in `run.json:artifacts` are valid
   relative paths from the run directory; absolute paths are never
   written.
7. **Schema versioning.** Every JSON schema file (`run.json`,
   `summary.json`, `_meta.json`) and the `GET /info` response
   carries a top-level `schema_version` string. The harness MUST
   write the version it implements; consumers SHOULD validate before
   parsing.
8. **Identifier naming consistency.** All `submit_NNN/` directory
   names (under both `checkpoints/` and `workspace/feedback/`) use
   zero-padded width `max(3, len(str(episode_budget)))`. All
   `ep_<XXX>/` directory names use the **same** width (since the
   maximum possible episode count equals the total episode budget).
   Episode indices are **run-global**: a single counter increments
   across submits, advancing only per successfully run episode (failed
   submits do not consume episode IDs). See `SPEC.md §4.0` for the
   indexing rules.
9. **Logs/feedback separation.** Files under `logs/` are never read
   by the agent (they live outside the workspace). Implementations
   MUST NOT route information intended for the agent's iteration
   loop into `logs/`; all such information goes into
   `workspace/feedback/`. Diagnostic information from policy
   execution (stdout, stderr, exceptions, timeouts) MUST be captured
   into the corresponding `feedback/submit_NNN/` files (`stdout.txt`,
   `stderr.txt`, `errors.txt`) and not into `logs/harness.log`.

Validation tooling (`hlbench check <run_dir>`) MUST verify all nine
invariants and report violations as harness bugs.

---

## 10. Out-of-Scope (For This Version)

- **Streaming partial run.json**: write happens once at end. For live
  monitoring, consume `workspace/feedback/submit_NNN/summary.json` as
  they appear.
- **Cross-run aggregation files**: there is no `runs/<model>/_index.json`
  or similar. Aggregation is the analysis tool's responsibility.
- **Symbolic links to shared artifacts**: snapshots are independent
  even if the harness uses hardlinks internally; the contract is that
  paths under `checkpoints/` resolve to the right contents.

---

## 11. Quick Reference

| Want to... | Read this file |
|---|---|
| Identify a run, get headline score | `run.json` |
| See what code the agent finally submitted | `workspace/system/` |
| See the per-submit code history | `checkpoints/submit_*/` |
| See the per-submit environmental feedback | `workspace/feedback/submit_*/` |
| Debug what went wrong in a failed run | `logs/harness.log` |
| Compute agent token/cost metrics | `logs/agent.jsonl` |
| Get the effective config the agent saw | call `GET /info` (server endpoint) |
| Get the final aggregate results | `run.json` |
| Get the raw 100 held-out returns | `run.json:outcome.held_out_returns` |
| See what the policy printed during an episode | `workspace/feedback/submit_*/episodes/ep_<XXX>/stdout.txt`, `stderr.txt` |
| Replay or backtest a specific episode | `workspace/feedback/submit_*/episodes/ep_<XXX>/trajectory.jsonl` (ep_<XXX> is run-global) |
