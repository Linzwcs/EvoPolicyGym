# docs/submit-protocol.md — Submission Protocol

This document specifies the behavior of a single submission: its
lifecycle, verdicts, resource accounting, and the rules that govern
how an agent interacts with the harness through the submit interface.

`SPEC.md §3` defines the *call* (CLI / Python API, side effects).
This document defines the *contract*: what the submit interface
guarantees about state transitions, failure modes, and resource
consumption.

---

## 1. Mental Model

A submission is **sealed code shipped to a judge, evaluated under
strict resource constraints, and returned with a verdict plus
diagnostic data**.

```
agent code (system/)
       │
       │ submit --env-instances <spec>
       ▼
   ┌────────────────────┐
   │  judge (harness)   │
   │   sandbox + run    │
   └────────────────────┘
       │
       │ verdict + diagnostics
       ▼
   feedback/submit_NNN/
       │
       │ agent reads, refines code
       ▼
   (next submission)
```

Key properties:

- **Sealed**: once submitted, `system/` is snapshotted; subsequent
  edits do not affect the in-flight submission.
- **Sandboxed**: execution respects all sandbox rules (`AGENTS.md §3`).
- **Verdict-based**: every submission ends with a verdict — one of
  the 10 enumerated values in §3 — plus diagnostic files.
- **Iterative**: unlike one-shot evaluation, submissions form a
  feedback loop. Agents read each verdict and refine.

Two important distinctions from a typical pass/fail judging system:

| Aspect | Typical judge | hlbench |
|---|---|---|
| Verdict | Pass/Fail (AC/WA) | Continuous score from held-out evaluation |
| Tests | Fixed input/output pairs | Stateful interactive environments |
| Interaction | One-shot per problem | Many submissions per task, iterative |
| Visible state | None (or sample tests) | Per-submission feedback + cumulative history |

---

## 2. Submission Lifecycle

Every submission progresses through seven phases. Failure at any
phase produces a specific verdict (see §3) and short-circuits the
remaining phases.

### 2.1 Phase diagram

```
                     ┌─────────────┐
                     │  Request    │  ◀── agent invokes submit --env-instances <spec>
                     └──────┬──────┘
                            │ valid? ──── no ──→ budget_invalid (no budget consumed)
                            │ yes
                            ▼
                     ┌─────────────┐
                     │  Snapshot   │  ◀── copy system/ to isolated location
                     └──────┬──────┘
                            │ size ok? ─── no ─→ oversize
                            │ yes
                            ▼
                     ┌─────────────┐
                     │  Validate   │  ◀── policy.py present? imports allowed?
                     └──────┬──────┘
                            │ ok? ──── no ─→ missing_policy | denied_import
                            │ yes
                            ▼
                     ┌─────────────┐
                     │  Compile    │  ◀── Python import of policy.py
                     └──────┬──────┘
                            │ ok? ──── no ─→ import_error
                            │ yes
                            ▼
                     ┌─────────────┐
                     │ Initialize  │  ◀── construct Policy(...)
                     └──────┬──────┘
                            │ ok? ──── no ─→ init_error | init_timeout
                            │ yes
                            ▼
                     ┌─────────────┐
                     │  Execute    │  ◀── run N episodes
                     └──────┬──────┘
                            │ ok? ──── no ─→ oom | submit_wall_exceeded
                            │ yes
                            │ (per-episode failures recorded in
                            │  ep_<XXX>/error.txt, do NOT change
                            │  submit-level verdict)
                            ▼
                     ┌─────────────┐
                     │   Commit    │  ◀── write feedback/submit_NNN/ atomically
                     └──────┬──────┘
                            ▼
                          [ok]
```

### 2.2 Phase-by-phase detail

#### Phase 1: Request

Agent invokes `hlbench submit --env-instances <spec>` (or Python
API). Harness expands the spec to a concrete list of integer IDs and
validates:

- Every ID is in `[0, n_env_instances)` (otherwise: verdict
  `invalid_env_instance`).
- `len(IDs) ≥ min_episodes_per_submit`
- `len(IDs) ≤ max_episodes_per_submit`
- `len(IDs) ≤ remaining_budget`
  (otherwise: verdict `budget_invalid`)

If any check fails, the submit is **rejected without consuming any
budget**. This is the **only** phase where rejection does not consume
budget — once we leave Phase 1, the agent has committed resources.

#### Phase 2: Snapshot

The harness recursively copies `system/` to an isolated location.
The snapshot becomes the authoritative version for this submission;
subsequent edits to `system/` do not affect the in-flight submit.

Validation: the snapshot's total size must not exceed
`system_total_bytes`; no individual file may exceed
`system_single_file_bytes`. See `AGENTS.md §3.3` for the size
calculation rule (source files only; `__pycache__/` etc. excluded).

Failure → `oversize`.

#### Phase 3: Validate

Static checks on the snapshot:

- `system/policy.py` exists at the top level.
- Static import scan of `policy.py` and any module reachable via
  imports finds no entries from the denied list.
- Module structure is valid Python.

Failure modes:
- `missing_policy` — no `system/policy.py` (file does not exist).
- `denied_import` — at least one denied module is imported.

#### Phase 4: Compile

Python imports `system.policy`. The harness sets `sys.path[0] =
system/` (see `SPEC.md §2.4`).

Failure → `import_error`. The error's `traceback` includes the full
Python import error (syntax error, missing module, circular import,
etc.).

#### Phase 5: Initialize

The harness constructs `Policy(obs_space, action_space, env_meta)`.
The construction is bounded by `policy_load_wall_s`.

Failure modes:
- `init_error` — `Policy.__init__` raised.
- `init_timeout` — `Policy.__init__` exceeded the wall-time limit.

Anything printed by `__init__` is captured and attached to the
**first episode's** `stdout.txt` (since `__init__` runs once per
submit, before episodes begin). If the submit fails at Phase 5, no
episode directory is created and any printed output is appended to
the submit-level `errors.txt` traceback's `message` field.

#### Phase 6: Execute

The harness runs `N` episodes, drawing seeds from the in-loop seed
pool. For each episode:

1. Create `episodes/ep_<global_id>/` directory.
2. Open `stdout.txt`, `stderr.txt` for capture.
3. Open `trajectory.jsonl` for append.
4. Open `observations.npy` (or `.npz`) if external obs storage.
5. Open `video.mp4` writer if env supports rendering.
6. Call `Policy.reset(episode_index)`.
7. Loop: `obs = env.step(action)`, record step, until terminated/truncated.
8. Close all files.

Per-episode failures (`reset_error`, `act_error`, `act_timeout`)
are recorded in `ep_<global_id>/error.txt` but do NOT change the
submit-level verdict; subsequent episodes still run.

Submit-level failures in Phase 6:
- `oom` — combined process RSS exceeded `submit_peak_rss_bytes`.
- `submit_wall_exceeded` — total wall time across all episodes
  exceeded `submit_wall_s`.

When a submit-level failure occurs during Phase 6, episodes
completed before the failure retain their full artifacts (with
global IDs assigned). The episode in progress at the moment of
failure has whatever artifacts were written before the failure;
the rest of the requested episodes are not attempted.

#### Phase 7: Commit

The harness aggregates per-episode artifacts into `summary.json`
and writes it. At this point, `feedback/submit_NNN/` is considered
complete and visible to the agent.

The harness uses a temp-file-plus-rename pattern for `summary.json`
to prevent the agent from observing a half-written file.

Verdict: `ok` (or one of the Phase 6 verdicts if Phase 6 failed).

### 2.3 Phase summary table

| # | Phase | What runs | Possible verdicts | Budget consumed if fail? |
|---|---|---|---|---|
| 1 | Request | parameter validation | `budget_invalid` | **No** |
| 2 | Snapshot | filesystem copy + size check | `oversize` | Yes |
| 3 | Validate | static import scan | `missing_policy`, `denied_import` | Yes |
| 4 | Compile | Python `import` | `import_error` | Yes |
| 5 | Initialize | `Policy.__init__` | `init_error`, `init_timeout` | Yes |
| 6 | Execute | run N episodes | `oom`, `submit_wall_exceeded` | Yes |
| 7 | Commit | write `summary.json` | (no agent-facing failure) | Yes |

Phases 1 vs 2 distinction is **critical**: rejection in Phase 1 is
the agent's request being malformed (not a code problem). Rejection
in Phases 2+ means the agent's code or behavior caused the failure,
and resources have been committed.

---

## 3. Verdict System

### 3.1 Complete enum

The `status` field of `summary.json` and the `category` field of
error entries use a unified enum of **11 verdicts**:

| Verdict | Phase | What it means |
|---|---|---|
| `ok` | 6 (success) → 7 | Submit ran all requested episodes |
| `budget_invalid` | 1 | Requested episode count outside `[min, max, remaining]` |
| `invalid_env_instance` | 1 | Requested env instance ID outside `[0, n_env_instances)` |
| `oversize` | 2 | Snapshot exceeded `system_total_bytes` or `system_single_file_bytes` |
| `missing_policy` | 3 | No `policy.py` or no `Policy` class at expected location |
| `denied_import` | 3 | Snapshot imports a forbidden module |
| `import_error` | 4 | Python `import` raised (syntax, missing module, circular, etc.) |
| `init_timeout` | 5 | `Policy.__init__` exceeded `policy_load_wall_s` |
| `init_error` | 5 | `Policy.__init__` raised |
| `oom` | 6 | Process RSS exceeded `submit_peak_rss_bytes` |
| `submit_wall_exceeded` | 6 | Total submit wall time exceeded `submit_wall_s` |

### 3.2 Per-episode error categories

Per-episode `error.txt` files use a separate (overlapping) enum for
events that fail individual episodes without failing the submit
itself:

| Category | When | Submit-level effect |
|---|---|---|
| `reset_error` | `Policy.reset()` raised | None; submit continues |
| `act_error` | `Policy.act()` raised | None; submit continues |
| `act_timeout` | `act()` exceeded `act_wall_ms` | None; submit continues |

These never appear in `summary.json:status` (which is submit-level).
They appear in `summary.json:errors` / `summary.json:timeouts`
(local index arrays) and the corresponding per-episode `error.txt`.

### 3.3 Verdict-to-feedback mapping

| Verdict | `summary.json` written? | `errors.txt` (submit-level) written? | `episodes/` exists? |
|---|---|---|---|
| `ok` | Yes (`status: "ok"`, full fields) | No | Yes |
| `budget_invalid` | Yes (minimal: status + remaining_budget) | Yes (one entry) | No |
| `invalid_env_instance` | Yes (minimal) | Yes (one entry) | No |
| `oversize` | Yes | Yes | No |
| `missing_policy` | Yes | Yes | No |
| `denied_import` | Yes | Yes | No |
| `import_error` | Yes | Yes | No |
| `init_timeout` | Yes | Yes | No |
| `init_error` | Yes | Yes | No |
| `oom` | Yes (partial: status + whatever episode metrics completed) | Yes | Yes (with completed episodes) |
| `submit_wall_exceeded` | Yes (partial) | Yes | Yes (with completed episodes) |

`oom` and `submit_wall_exceeded` are special: they occur during
Phase 6, so some episodes may have completed normally before the
failure. Their `episodes/` directory contains those completed episodes,
and the submit-level `errors.txt` describes the failure. This is the
only situation in which both `episodes/` and submit-level `errors.txt`
coexist; otherwise they are mutually exclusive.

> **Note**: the mutual exclusion stated in `SPEC.md §4.4.4` is
> *almost* absolute, with `oom` and `submit_wall_exceeded` as the two
> exceptions. Validation tools MUST allow both files to coexist when
> `status ∈ {oom, submit_wall_exceeded}`.

---

## 4. Resource Accounting

### 4.1 Budget consumption

Episode budget is consumed at the start of Phase 2 (Snapshot) — the
moment the harness commits resources beyond mere parameter parsing.
A rejected Phase 1 (the only "free" rejection) does not consume any
budget.

Once consumed, the full requested `N` is gone, regardless of how
many episodes actually ran:

- Phase 2–5 failures: 0 episodes ran, but `N` was committed → `N`
  consumed.
- Phase 6 partial completion (oom / wall_exceeded): some episodes
  ran, but the agent committed to `N` upfront → `N` consumed.
- Phase 6 full success: `N` episodes ran → `N` consumed.

This rule is intentionally simple: **once the snapshot is taken,
the agent owes the full requested count**. Agents that want to
limit downside should submit smaller batches.

### 4.2 Atomicity

Within a submit, `feedback/submit_NNN/summary.json` is written via
temp-file-plus-rename, so the agent never reads a partial
`summary.json`.

Other files in `feedback/submit_NNN/` may be appended during Phase 6
(specifically `episodes/ep_<XXX>/trajectory.jsonl` and the stdout /
stderr captures). They are finalized before the harness writes
`summary.json` — the presence of `summary.json` indicates that all
other files in the same submit directory are complete.

**Recommended consumer pattern**: poll for `summary.json`; once
present, treat the entire directory as ready.

### 4.3 Sequential execution

The server processes **one submit at a time**. Concurrent submits
from the agent are not supported:

- `POST /submit` (HTTP) blocks until the submit completes.
- Issuing a second `POST /submit` before the first returns is
  undefined behavior (implementations MAY queue, reject with
  HTTP 409, or crash).

Rationale: the iterative refinement model requires the agent to read
each verdict before deciding the next move. Concurrent submits would
either (a) be redundant (same code under different seeds), (b)
require speculative branching, or (c) imply parallel evaluation —
none of which match the protocol's intent.

### 4.4 No automatic deduplication

The harness does **not** detect identical re-submissions. If the
agent submits the same `system/` snapshot twice, both submits run
normally and both consume budget.

Rationale:
- Agents may legitimately re-run identical code to assess variance.
- Detecting "identical" requires defining file equality across
  ordering, mtime, etc. — complexity without clear benefit.
- Agents that want dedup behavior can hash their own snapshots and
  skip submission themselves.

### 4.5 No rate limiting

Submits can be issued back-to-back as fast as the harness can
process them. There is no minimum interval between submits.

Rationale: budget is the limiter. Rate limits would add a second
axis with no benefit — the budget consumption already prevents
spam.

---

## 5. Iterative Refinement Model

This is the part of the protocol that **differs from a typical
one-shot judge**: agents are expected to submit, read feedback,
refine, and submit again.

### 5.1 The refinement loop

```
   ┌────────────────────────────────────────────────┐
   │                                                │
   │   1. Edit code in system/                      │
   │   2. Submit (consumes N episodes from budget)  │
   │   3. Read feedback/submit_NNN/                 │
   │      - summary.json (verdict + aggregates)     │
   │      - episodes/ep_<XXX>/* (per-episode detail)│
   │      - errors.txt / error.txt if failures      │
   │   4. Decide: refine code? change strategy?     │
   │      keep submit?                              │
   │   5. (back to 1)                               │
   │                                                │
   └────────────────────────────────────────────────┘
                            │
                            │ budget exhausted, or
                            │ agent declares finished
                            ▼
                  ┌──────────────────────┐
                  │  Held-out evaluation │  (one-time, by harness)
                  └──────────────────────┘
                            │
                            ▼
                       final_score
```

### 5.2 Recommended cadence

Strategic budget allocation is itself a tested capability. Common
cadence patterns:

| Pattern | Episodes per submit | When to use |
|---|---|---|
| Cheap probe | 1–3 | Sanity-check a code change quickly |
| Standard eval | 8 | Default; balances signal vs. cost |
| High-confidence | 16–32 | Late in the run; choose between candidates |
| Burn-budget | rest of budget | Last submit, fully exploit remaining |

Agents are not required to follow any particular cadence; the
budget consumption rule (§4.1) makes the cost of each pattern
transparent.

### 5.3 Reading feedback to decide next move

A submit's `summary.json` exposes:

- `status` — did the submit run? (the verdict)
- `mean_return` / `std_return` / `min_return` / `max_return` — return distribution
- `episode_lengths` — per-episode step counts (variable-length envs)
- `timeouts` / `errors` — which local episodes had issues
- `reward_components_per_episode` — sub-objective breakdown (if env declares)
- `submit_started_at` / `submit_completed_at` — timing (for self-assessment)
- `remaining_budget` — how much budget is left

For deeper diagnosis, the per-episode files
(`trajectory.jsonl`, `stdout.txt`, `video.mp4`, `error.txt`)
provide step-level granularity.

The agent is the sole decision-maker for what to do next. The
harness provides data, not advice.

---

## 6. Anti-Cheating Provisions Specific to Submit

The full anti-hack rules are in `AGENTS.md §5`. This section calls
out submit-related provisions specifically.

### 6.1 Snapshot isolation

The snapshot taken in Phase 2 is byte-identical to `system/` at the
moment of submission. The agent cannot modify the snapshot after
submit. This prevents:

- Time-of-check-to-time-of-use attacks (edit code between validation
  and execution).
- Code that mutates itself in response to feedback within a single
  submit.

### 6.2 No external state during execution

During Phase 6, the policy cannot:

- Open network connections (sandbox enforces).
- Read files outside `system/` (sandbox enforces).
- Spawn subprocesses or threads beyond the harness's main process
  (sandbox enforces).
- Persist state to anywhere other than `system/` files.

The only persistence channel across submits is the `system/`
directory.

### 6.3 No held-out seed access

The harness ensures that:

- The held-out seed pool is never written to any file inside
  `workspace/` or `runs/`.
- Environment variables visible to the policy do not contain seed
  values.
- The policy cannot enumerate or guess the held-out pool from
  `GET /info` — held-out size, seeds, and results are not exposed
  there or anywhere else reachable by the agent.

### 6.4 Budget is committed at Snapshot

An agent cannot "test the waters" by submitting a syntactically
broken policy to discover validation rules without paying the
budget. Phase 2 commitment ensures every submit that gets past
Phase 1 pays for itself.

---

## 7. Validation Requirements for `hlbench check`

The benchmark's validation tool MUST verify the following for each
submit's `feedback/submit_NNN/` directory:

### 7.1 Verdict consistency

- `summary.json:status` is one of the 10 verdicts in §3.1.
- If `status == "ok"`: `episodes/` exists, no submit-level
  `errors.txt`, `n_episodes` directories under `episodes/`.
- If `status` ∈ {`oom`, `submit_wall_exceeded`}: both `episodes/`
  and `errors.txt` may exist; `episodes/` contains 0 to `n_episodes`
  fully-formed episode directories.
- Otherwise (`budget_invalid`, `oversize`, etc.): `errors.txt`
  exists, `episodes/` does NOT exist.

### 7.2 Budget accounting

- Sum of `n_episodes` across all submits with `status != "budget_invalid"`
  equals `episode_budget - run.json:outcome.auxiliary.remaining_budget_final`.
- `remaining_budget` is monotonically non-increasing across submits.

### 7.3 Per-episode invariants

- See `SPEC.md §4.8.4` F1–F9 for per-episode file invariants
  (trajectory length, observations.npy shape, video frame count, etc.).

### 7.4 Submit ordering

- `submit_NNN` directories are numbered contiguously from 0.
- `summary.json:submit_index` matches the directory name.

Failures of any of these checks indicate a harness bug, not an
agent issue.

---

## 8. Quick Reference

### 8.1 Verdict cheat sheet

```
ok                     → submit ran, episodes executed (success)
budget_invalid         → bad request count, no resources consumed
invalid_env_instance   → ID out of [0, n_env_instances), no resources consumed
oversize               → system/ too big
missing_policy         → no policy.py / no Policy class
denied_import          → forbidden import detected
import_error           → Python import failed
init_timeout           → Policy.__init__ too slow
init_error             → Policy.__init__ raised
oom                    → memory exceeded during execution
submit_wall_exceeded   → total wall time exceeded during execution
```

### 8.2 "I got verdict X, what should I do?"

| Verdict | Next move |
|---|---|
| `ok` | Read `summary.json`, decide refinement |
| `budget_invalid` | Reduce `--env-instances` count; submit again (no cost) |
| `invalid_env_instance` | Use IDs in `[0, n_env_instances)`; submit again (no cost) |
| `oversize` | Trim `system/` (delete cached files, simplify code) |
| `missing_policy` | Restore `system/policy.py` with valid `Policy` class |
| `denied_import` | Replace the denied module with an allowed one |
| `import_error` | Fix the import / syntax error (read `errors.txt` for the traceback) |
| `init_timeout` | Move heavy work out of `__init__`, or split into smaller `system/` reads |
| `init_error` | Fix the exception in `__init__` |
| `oom` | Reduce memory footprint (smaller arrays, free intermediate state) |
| `submit_wall_exceeded` | Speed up `act()` or reduce planning depth |

### 8.3 Budget consumption rules at a glance

```
Phase 1 (Request)        rejection → 0 episodes consumed
Phase 2+ (Snapshot..)    rejection → N episodes consumed (full request)
Phase 6 partial          partial   → N episodes consumed (full request)
Phase 6 full + Phase 7   success   → N episodes consumed
```

**Rule of thumb**: once the harness takes a snapshot, the cost is
locked in. The only "free" rejection is malformed parameters
(Phase 1).
