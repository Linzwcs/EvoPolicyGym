# SPEC.md — Technical Specification

This document specifies the contracts between agents, policies, and
the hlbench-pro harness. `AGENT.md` defines what agents may do; this
document defines how the system actually works.

---

## 1. Workspace Layout (Normative)

Each env ships a starter workspace, automatically materialized by
the harness at run start. A benchmark run materializes this directory
and the agent and harness share it.

```
workspace/
├── TASK.md              # human-readable task description, delivered by server at run start
├── AGENT.md             # protocol rules (client-side, shipped with agent harness)
├── system/              # agent-writable Python package; contains policy.py at minimum
│   ├── policy.py        # required entry point (top level)
│   ├── controllers/     # example: agent-organized submodule
│   │   ├── __init__.py
│   │   ├── pid.py
│   │   └── mpc.py
│   ├── utils/
│   │   └── filters.py
│   ├── memory/          # example: persistent state across submits
│   │   └── stats.json
│   ├── tests/           # example: agent-authored self-tests (agent runs them itself)
│   │   └── test_pid.py
│   └── .final_submit    # optional: agent designates a specific submit as final
└── feedback/            # populated by server directly into shared workspace
    ├── submit_000/
    ├── submit_001/
    └── ...
```

The agent harness creates `feedback/` if it does not exist. It never
modifies `system/`. The agent never modifies `feedback/`, `TASK.md`,
or `AGENT.md`.

**Workspace contents are local to the agent's machine.** Server
state (config, env metadata, feedback artifacts) reaches the
workspace via network transport from the per-run server. The
workspace is essentially a local mirror of the agent's perspective:

| File / dir | Source | When populated |
|---|---|---|
| `TASK.md` | Server | At run start, fetched once |
| `AGENT.md` | Agent harness | At run start (shipped with harness) |
| `system/` | Agent (own writes) | Throughout the run |
| `feedback/submit_NNN/` | Server (writes directly via shared FS) | After each submit returns |

**Dynamic state** (remaining budget, current submit status, etc.) is
not persisted as a file — it is fetched on demand via `GET /info`
(see §1.1).

### 1.1 `GET /info` — Effective Run Config

The server provides current run configuration via the `GET /info`
endpoint (no workspace file). The response is a single JSON
document containing both static config (set at run start) and
dynamic state (refreshed every call). Agents call this at run start
to learn the run's constraints, and again whenever they need
up-to-date dynamic state (notably `remaining_budget`).

```json
{
  "schema_version": "0.1",

  "env": "halfcheetah",
  "env_version": "0.1",
  "harness_version": "0.1.0",
  "agent_md_hash": "sha256:...",

  "episode_budget": 256,
  "min_episodes_per_submit": 1,
  "max_episodes_per_submit": 256,

  "resource_limits": {
    "system_total_bytes": 51200,
    "system_single_file_bytes": 25600,
    "act_wall_ms": 10,
    "policy_load_wall_s": 1,
    "submit_wall_s": 300,
    "submit_peak_rss_bytes": 1073741824
  },

  "allowed_imports": ["numpy", "scipy", "math", "..."],
  "denied_imports": ["transformers", "huggingface_hub", "..."],

  "env_meta": {
    "obs_space": { "type": "Box", "shape": [17], "low": -inf, "high": inf },
    "action_space": { "type": "Box", "shape": [6], "low": -1.0, "high": 1.0 },
    "max_episode_steps": 1000,
    "n_env_instances": 256,
    "obs_storage": "inline",
    "reward_components": {
      "forward": "reward_forward",
      "ctrl": "reward_ctrl",
      "contact": "reward_contact"
    }
  },

  "state": {
    "remaining_budget": 248,
    "n_submits": 3,
    "n_successful_submits": 2,
    "last_submit_index": 2,
    "last_submit_status": "ok",
    "submit_in_progress": false,
    "in_progress_submit_id": null,
    "is_finalized": false,
    "started_at": "2026-05-28T10:00:00Z"
  }
}
```

**Field categories:**

| Group | Fields | Changes during run? |
|---|---|---|
| Identity | `env`, `env_version`, `harness_version`, `agent_md_hash` | Never |
| Budget rules | `episode_budget`, `min/max_episodes_per_submit` | Never |
| Resource limits | `resource_limits.*` | Never |
| Import policy | `allowed_imports`, `denied_imports` | Never |
| Env metadata | `env_meta.*` | Never |
| Dynamic state | `state.*` | Yes (on every successful submit) |

**Deliberately not exposed to the agent:**

- **Held-out evaluation parameters** (size, seeds, results): the
  held-out pool is server-internal. Agents have no visibility into
  it at any point — during or after the run.
- **`expert_baseline` and `random_baseline`**: env-internal scoring
  references. Agents see raw `mean_return` values; they do not see
  what "expert level" or "random level" means in numerical terms.
  This forces agents to optimize without targeting a known
  threshold.
- **Real seed values**: agents address env instances by integer ID
  `[0, n_env_instances)`; the mapping to real seeds is server-side.

`env_meta.n_env_instances` is the count of distinct env instances
available for the agent to submit against. Each is a deterministic
environment with a hidden real seed; the mapping from instance ID to
real seed is loaded from the env's static `train.json` file
(env-internal, never exposed). Submitting to an out-of-range ID
returns the `invalid_env_instance` verdict (see §4.1).

`env_meta.reward_components` is populated automatically from the
environment definition. The environment author declares which `info`
dict keys constitute reward components as part of the env's
registration; the harness reads this declaration and extracts those
keys at every step, exposing them in `feedback/` (see §4.1, §4.2).
When the environment declares no components (e.g., Atari, CartPole),
this field is absent or empty and only the total reward is reported.

`env_meta.obs_storage` controls how observations are stored in the
per-episode feedback (see §4.2 and §4.6):

- `"inline"` (default): each `obs` field in `trajectory.jsonl` is the
  observation serialized as JSON nested lists. The harness rejects
  envs whose serialized `obs` exceeds 10 KB.
- `"external"`: `obs` in `trajectory.jsonl` is `null`; observations
  are stored in a side-car binary file `observations.npy` in the
  same `ep_<XXX>/` directory. Used by pixel-based envs (CarRacing,
  pixel Atari, MuJoCo camera) where inline storage would explode.

**Submit/episode directory width** is derived as
`max(3, len(str(episode_budget)))`. Agents that need it should
compute it themselves; it is not exposed as a separate field.

**Field provenance**: every value comes from merging (a) the
environment registration's defaults and (b) run-time overrides
passed to `hlbench run`. The agent does not see which value came
from which source; the merged result is the truth.

Agents typically:
- Call `GET /info` once at run start to read static config and
  `env_meta`.
- Call `GET /info` after each submit (or periodically) to refresh
  `state` — most importantly `remaining_budget`.

---

## 2. Policy Interface

`system/policy.py` MUST define a class named `Policy` with this
contract:

```python
import numpy as np
from typing import Any, Mapping

class Policy:
    def __init__(
        self,
        obs_space: "gymnasium.Space",
        action_space: "gymnasium.Space",
        env_meta: Mapping[str, Any],
    ) -> None:
        """Constructed once per submit (i.e., shared across the n_episodes
        of a single submit). May read files from system/. Must not read
        from feedback/, TASK.md, or anywhere outside the workspace.

        env_meta contains fields documented in TASK.md and exposed via
        `GET /info:env_meta`, plus:
            - env: str                   (environment slug, e.g., "halfcheetah")
            - submit_index: int          (0-based, increments per submit)
            - n_episodes_this_submit: int (how many episodes this submit will run)
            - remaining_budget_after: int (budget after this submit completes)
            - max_episode_steps: int
            - allowed_imports: tuple[str, ...]   (informational)
        """
        ...

    def reset(self, episode_index: int) -> None:
        """Called at the start of every episode in the submit. episode_index
        ranges over [0, episodes_per_submit). The episode seed is NOT
        passed and MUST NOT be inferred or guessed."""
        ...

    def act(self, obs) -> "action":
        """Called once per step. Must return an action valid for
        action_space. Wall-time limit applies per call (default 10 ms;
        see AGENT.md §3.3)."""
        ...

    def on_episode_end(self, episode_return: float) -> None:
        """Optional. Called once at the end of each episode with the
        episode's total undiscounted return. May update internal state
        (e.g., running statistics in system/)."""
        ...
```

### 2.1 State Persistence

- **Within a submit:** instance attributes of `Policy` persist across
  the `n_episodes_this_submit` episodes. `__init__` is called once;
  `reset()`/`act()`/`on_episode_end()` are called per episode.
- **Across submits:** the `Policy` instance is destroyed after each
  submit. To carry state across submits, write to `system/` files
  inside `on_episode_end()` or by structuring `__init__` to read from
  `system/`.

### 2.2 Action and Observation Spaces

Spaces follow the [Gymnasium](https://gymnasium.farama.org/) convention.
Observations and actions are `numpy.ndarray` or `int`/`float` scalars
as appropriate. Discrete actions are returned as `int`; continuous
actions as a 1-D `numpy.ndarray` of correct shape and dtype.

### 2.3 Errors

**At submit setup** (before any episode runs):

- If `Policy.__init__` raises, the submit fails at the submit level:
  `summary.json:status = "init_error"`, no episodes run,
  `feedback/submit_NNN/errors.txt` records the traceback as a JSON
  Lines entry (see §4.4), and the submit's full requested episodes
  count against the budget.
- If `Policy.__init__` exceeds `policy_load_wall_s`, same as above
  but with `status = "init_timeout"`.

**During an episode**:

- If `Policy.reset()` raises, that episode fails immediately:
  `episodes/ep_<XXX>/error.txt` records the traceback with
  `category: "reset_error"`. `trajectory.jsonl` for that episode is
  empty (no steps were taken). The episode still counts as one
  toward `n_episodes` and the global counter still advances.
  Subsequent episodes in the same submit proceed with fresh
  `reset()` calls.
- If `act()` raises mid-episode, that episode terminates with the
  return accumulated so far. The failing step's recorded `action`
  is the harness's fallback (`action_space.sample()`), allowing
  the trajectory to remain complete up to and including the failing
  step. `error.txt` records the traceback with
  `category: "act_error"`. Subsequent episodes in the same submit
  proceed with a fresh `reset()`.
- If `act()` exceeds `act_wall_ms`, the action defaults to
  `action_space.sample()`, the episode continues, and an entry with
  `category: "act_timeout"` is appended to `error.txt`. Repeated
  timeouts within the same episode are all logged.
- If `on_episode_end()` raises, an entry with
  `category: "on_episode_end_error"` is recorded but the episode's
  return and other artifacts are preserved.

In all cases, each error becomes one line in the appropriate file
using the JSON Lines schema defined in §4.4.

### 2.4 Python Path and Imports

`workspace/system/` is the agent's Python package. The harness
prepares the import environment as follows:

- **`sys.path[0] = workspace/system/`** when `policy.py` is loaded.
  All other paths in `sys.path` come from the standard Python install
  (allowed third-party libraries per §3.2).
- **`policy.py` MUST be at `system/policy.py`** (top level of the
  package). Other modules can be placed anywhere under `system/`.
- **Imports from agent-authored modules** use standard Python
  resolution from `system/`:
  ```python
  # In system/policy.py:
  from controllers.pid import PIDController
  from utils.filters import lowpass
  import json
  ```
- **Both regular packages** (with `__init__.py`) and **namespace
  packages** (without) are supported. Either works.
- **Imports of third-party libraries** (numpy, scipy, etc.) are
  subject to the allow/deny lists in §3.2. Imports between
  agent-authored modules under `system/` are not subject to those
  lists.
- **Agents MUST NOT** modify `sys.path` at runtime to escape the
  workspace, import compiled binaries from outside `system/`, or
  manipulate the import system to bypass the sandbox.

The harness also sets the working directory (`os.getcwd()`) to
`workspace/system/` for the duration of `Policy.__init__`,
`reset()`, `act()`, and `on_episode_end()`. This means relative
file paths inside agent code resolve against `system/`. Agents
needing to read their own data files SHOULD use either:

```python
# Option A: relative to cwd (which equals system/)
with open("memory/stats.json") as f: ...

# Option B: relative to module file (more robust to cwd changes)
import os
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "memory/stats.json")) as f: ...
```

Both work; Option B is recommended for robustness.

---

## 3. Server Interface

The per-run server exposes three HTTP endpoints. Agents control runs
through these endpoints; per-submit feedback artifacts are read from
the shared `workspace/feedback/` directory (server writes directly).

| Endpoint | Method | Purpose |
|---|---|---|
| `/info` | GET | Effective config + dynamic state |
| `/submit` | POST | Run a synchronous submit |
| `/finalize` | POST | Trigger held-out evaluation |

For implementation convenience the same operations are also
exposed as a Python library (`hlbench.core.Server`) for tests and
internal tooling. **Agents must use HTTP**; the lib is not an
agent-facing interface. The HTTP layer is a thin FastAPI wrapper
over the lib.

For the **submission lifecycle, verdict system, and behavioral
contract**, see `docs/submit-protocol.md`.

### 3.1 `GET /info`

Returns the JSON schema described in §1.1. Static fields are stable
for the run; `state.*` fields refresh on every call.

```http
GET /info HTTP/1.1
Accept: application/json
```

Response body: see §1.1 for schema.

### 3.2 `POST /submit`

Synchronously runs the requested env instances and writes feedback
files to `workspace/feedback/submit_NNN/`. Blocks until done.

```http
POST /submit HTTP/1.1
Content-Type: application/json

{ "env_instances": [0, 1, 2, 3, 4, 5, 6, 7] }
```

`env_instances` must be a JSON array of integer IDs. Each ID in the
list consumes one episode from the budget; re-submitting the same ID
within one request (e.g., `[5, 5, 5]`) is allowed and produces three
independent episode runs.

The list is validated:
- Every ID must be in `[0, n_env_instances)`
  (`GET /info:env_meta.n_env_instances`). Otherwise: HTTP 400, response
  body `{ "status": "invalid_env_instance", ... }`. **No budget consumed.**
- `len(list)` must satisfy `min_episodes_per_submit ≤ len ≤
  min(max_episodes_per_submit, remaining_budget)`. Otherwise: HTTP 400,
  response body `{ "status": "budget_invalid", ... }`. **No budget consumed.**

Successful response:

```json
{
  "submit_id": 5,
  "status": "ok",
  "summary": {
    "schema_version": "0.1",
    "submit_index": 5,
    "env_instances": [0, 1, 2, 3, 4, 5, 6, 7],
    "n_episodes": 8,
    "first_global_episode": 40,
    "remaining_budget": 200,
    "returns": [...],
    "...": "..."
  }
}
```

The `summary` object is **byte-identical** to
`workspace/feedback/submit_005/summary.json` (which the server has
already written). Agents may use either — whichever is more
convenient.

For per-episode artifacts (trajectory, video, observations, stdout,
stderr, error), read them from
`workspace/feedback/submit_<submit_id>/episodes/ep_<XXX>/` directly.

### 3.3 `POST /finalize`

Declares the run finished and triggers held-out evaluation. Subsequent
submits raise an error.

```http
POST /finalize HTTP/1.1
```

Response:

```json
{ "status": "evaluating" }
```

Held-out evaluation runs synchronously after this call returns; the
final score is written to `runs/<model>/<env>/<exp-id>/run.json`
(see `docs/output.md §3`). The agent has no access to held-out
results.

If the budget is fully exhausted (`remaining_budget == 0`), the
server MAY auto-finalize without an explicit `POST /finalize` call.
This is implementation-defined; agents that want explicit control
should call `/finalize` themselves.

### 3.4 Python library API (internal, for tests and tooling)

Tests and harness internals may call the server lib directly:

```python
from hlbench.core import Server

server = Server(env_id="halfcheetah", workspace_dir="...")
info = server.info()
result = server.submit(env_instances=[0, 1, 2, ..., 7])
final = server.finalize()
```

This API is **not for agents**. Benchmark runs whose agents bypass
HTTP are not valid.

### 3.5 Side Effects of a Submit

When `POST /submit` is invoked, the server performs these steps in
order:

1. Validate the `env_instances` list:
   - Every ID in `[0, n_env_instances)` → otherwise reject with
     `invalid_env_instance` (HTTP 400, no budget consumed).
   - Count satisfies budget bounds → otherwise reject with
     `budget_invalid` (HTTP 400, no budget consumed).
2. Snapshot `system/` (recursive copy to an isolated location).
3. Validate the snapshot: size limit, import scan, `Policy` class
   present and importable.
4. If validation fails, write `feedback/submit_NNN/errors.txt` and
   `feedback/submit_NNN/summary.json` with the appropriate non-`ok`
   status (see §4.1 for the enum), then **decrement `remaining_budget`
   by the number of requested env instances**. The full committed
   budget is consumed even on validation failure.
5. Otherwise, instantiate `Policy` and run one episode per requested
   env instance (each using its associated hidden real seed loaded
   from the env's `train.json`).
6. Write `feedback/submit_NNN/` (see §4).
7. Decrement `remaining_budget` by the number of env instances run.
8. Return HTTP 200 with `{"submit_id", "status", "summary"}` body.

A submit is atomic from the agent's perspective: either step 1
rejects it (HTTP 400, no budget consumed) or steps 2–8 happen as a
unit (HTTP 200, full requested count consumed).

---

## 4. Feedback Format

After submit N, the harness writes:

```
feedback/submit_NNN/
├── summary.json                 (submit-level aggregate)
├── errors.txt                   (only if submit-level failure: validation/import/init)
└── episodes/
    ├── ep_<XXX>/                (XXX = run-global episode index, zero-padded)
    │   ├── trajectory.jsonl     (per-step data)
    │   ├── observations.npy     (only if env_meta.obs_storage == "external")
    │   ├── video.mp4            (only if env supports rendering)
    │   ├── stdout.txt           (captured policy stdout during this episode)
    │   ├── stderr.txt           (captured policy stderr during this episode)
    │   └── error.txt            (only if act() raised or timed out in this episode)
    ├── ep_<XXX+1>/
    │   └── ...
    └── ...
```

If a submit fails at the submit level (snapshot validation, import,
or `Policy.__init__` failure), no episodes run and no `episodes/`
directory is created. The failure is recorded in `summary.json:status`
and described in `errors.txt`.

### 4.0 Episode indexing

**Episode index `XXX` is unique across the entire run, not per
submit.** This enables direct references like "ep_142" without
ambiguity, which is essential for replay tools, backtests, and
cross-submit joins.

Rules:

- Width is the same as `submit_NNN` width:
  `max(3, len(str(episode_budget)))`. Derived from `episode_budget`
  (visible via `GET /info`).
- The counter advances by 1 **per episode that the harness attempted
  to run**, regardless of whether it succeeded or failed mid-flight.
  An "episode" is "one `reset()`-to-termination cycle attempted by
  the harness", not "one successful completion."
- A submit-level failure (no episodes attempted) does NOT advance
  the counter; the next successful submit picks up at the same
  global index.
- Within a single submit, episode indices are contiguous: a
  successful submit running `n_episodes` consumes global IDs
  `[first_global_episode, first_global_episode + n_episodes - 1]`.
- Episode global IDs and submit global IDs are independent counters.
  They happen to share the same width derivation; their values do
  not align in general.

**Edge cases**:

| Situation | Episode counter advances? | `episodes/ep_<XXX>/` created? |
|---|---|---|
| Episode runs to natural termination | yes | yes (full artifacts) |
| Episode hits `max_episode_steps` (truncated) | yes | yes (full artifacts) |
| `act()` raises mid-flight | yes | yes (artifacts up to failure + `error.txt`) |
| `act()` times out mid-flight | yes | yes (artifacts up to failure + `error.txt`) |
| `Policy.reset()` raises at episode start | yes | yes (empty `trajectory.jsonl` + `error.txt`) |
| `Policy.__init__` raises (submit-level) | **no** | no — no `episodes/` directory exists |
| Snapshot validation fails (e.g., `oversize`) | **no** | no — no `episodes/` directory exists |
| Submit's `act()` exceeds `submit_wall_s` mid-flight | yes (for episodes that ran) | yes (full or partial per episode) |

**Within-submit partial execution**: if a submit requests 8 episodes
and the 5th episode crashes via `act_error`, episodes 6–8 still run
on fresh `reset()` calls (per §2.3). All 8 get global IDs; episode
5's folder has `error.txt`, others don't. `summary.json:returns`
has length 8 and `errors: [4]` (the local index of the failure).

**Within-submit harness-level abort**: if the harness itself crashes
or hits the per-submit wall-time limit mid-flight, episodes that
completed before the crash retain their artifacts and global IDs.
The run's `outcome.status` becomes `error` (see §7), and
`summary.json` for that submit may be missing some fields or absent
entirely (see §4.7).

### 4.1 `summary.json` Schema

```json
{
  "schema_version": "0.1",

  "submit_index": 0,
  "env": "halfcheetah",
  "status": "ok",

  "n_episodes": 8,
  "first_global_episode": 0,
  "env_instances": [5, 100, 100, 7, 200, 5, 42, 3],
  "remaining_budget": 248,

  "submit_started_at": "2026-05-28T10:01:23.456Z",
  "submit_completed_at": "2026-05-28T10:02:11.234Z",
  "wall_time_seconds": 47.8,

  "returns": [1234.5, 1180.2, 1310.0, 1255.7, 1198.4, 1330.9, 1241.0, 1265.3],
  "mean_return": 1252.0,
  "std_return": 54.6,
  "min_return": 1180.2,
  "max_return": 1330.9,

  "episode_lengths": [1000, 1000, 873, 1000, 1000, 1000, 1000, 956],
  "mean_episode_length": 978.6,

  "timeouts": [],
  "errors": [2],

  "reward_components_mean": {
    "forward": 1480.2,
    "ctrl": -180.1,
    "contact": -48.1
  },
  "reward_components_per_episode": {
    "forward": [1502.3, 1410.7, ...],
    "ctrl":    [-178.4, -185.2, ...],
    "contact": [-50.1, -45.3, ...]
  }
}
```

`status` is one of the following unified values (same enum as
`_meta.json:validation_status` in `docs/output.md §5.1`):

| Value | Meaning |
|---|---|
| `ok` | Submit ran and episodes executed |
| `budget_invalid` | Rejected: requested episodes outside `[min, max, remaining]` |
| `oversize` | Rejected: snapshot exceeded `system_total_bytes` |
| `missing_policy` | Rejected: no `policy.py` or no `Policy` class |
| `denied_import` | Rejected: snapshot imported a forbidden module |
| `import_error` | Rejected: snapshot import raised (syntax, missing module, etc.) |
| `init_timeout` | Rejected: `Policy.__init__` exceeded `policy_load_wall_s` |
| `init_error` | Rejected: `Policy.__init__` raised |
| `invalid_env_instance` | Rejected: requested env instance ID outside `[0, n_env_instances)` |
| `oom` | Failed during execute: process RSS exceeded `submit_peak_rss_bytes` |
| `submit_wall_exceeded` | Failed during execute: total wall time exceeded `submit_wall_s` |

For the full submission lifecycle, phase-to-verdict mapping, and
mutual exclusion rules, see `docs/submit-protocol.md §3`.

### 4.1.1 Field-by-field

| Field | Type | Notes |
|---|---|---|
| `submit_index` | int | 0-based; matches the directory name `submit_NNN/` |
| `env` | string | Environment slug |
| `status` | string | See enum above |
| `n_episodes` | int | Episodes requested (and consumed from `remaining_budget`). Equals `len(env_instances)`. Equals the number of `ep_<XXX>/` directories created **on success**; on failure no episodes ran but the budget is still consumed |
| `first_global_episode` | int \| null | Run-global ID of `returns[0]`'s episode. `null` when `status != "ok"` (no episodes ran) |
| `env_instances` | array[int] | The env instance IDs the agent requested for this submit, in the order they were run. Length equals `n_episodes` |
| `remaining_budget` | int | Episode count remaining after this submit |
| `submit_started_at` | ISO-8601 string | When the harness began processing this submit |
| `submit_completed_at` | ISO-8601 string | When the harness finished writing this `summary.json` |
| `wall_time_seconds` | float | Convenience: `submit_completed_at - submit_started_at` |
| `returns` | array[float] \| null | Length `n_episodes`. `returns[i]` is the undiscounted return of `episodes/ep_<first_global_episode + i>/`, which used env instance `env_instances[i]`. `null` when `status != "ok"` |
| `mean_return`, `std_return`, `min_return`, `max_return` | float \| null | Aggregates of `returns`. `null` when `status != "ok"` |
| `episode_lengths` | array[int] \| null | Length `n_episodes`. Per-episode step count. `null` when `status != "ok"` |
| `mean_episode_length` | float \| null | Aggregate. `null` when `status != "ok"` |
| `timeouts` | array[int] \| null | Local indices into `returns` for episodes that hit an `act()` wall-time overrun. Empty array if none. `null` when `status != "ok"` |
| `errors` | array[int] \| null | Local indices into `returns` for episodes that raised an exception mid-flight. Empty array if none. `null` when `status != "ok"` |
| `reward_components_mean` | object \| null | Present only when `env_meta.reward_components` is non-empty AND `status == "ok"`. See §1.1 |
| `reward_components_per_episode` | object \| null | Per-component arrays of length `n_episodes`. Same conditions as above |

### 4.1.2 Conventions

**Indexing.** `returns`, `episode_lengths`, and the values inside
`reward_components_per_episode` are all parallel arrays indexed by
local position `i ∈ [0, n_episodes)`. The episode at position `i`
has global ID `first_global_episode + i` and lives at
`episodes/ep_<first_global_episode + i>/`.

`timeouts` and `errors` use the same local indexing. To get the
global ID, add `first_global_episode`. To get the trajectory folder
of the third episode that errored, do
`episodes/ep_<first_global_episode + errors[2]>/`.

**Failure semantics.** When `status != "ok"`, no episodes run and
no `episodes/` directory is created (see §4). All array fields and
their aggregates are `null` (not `[]`), making "did this submit
run?" a simple `status == "ok"` check. `n_episodes` and
`remaining_budget` still reflect the requested-and-consumed count.

**`reward_components_per_episode` keys.** Each key matches a name
declared in `env_meta.reward_components`. The value is a float
array of length `n_episodes` representing each episode's summed
component return.

### 4.2 `episodes/ep_<XXX>/trajectory.jsonl` Schema

`XXX` is the global episode index (see §4.0). One JSON object per
line; each line is one step of the episode (one transition: state →
action → reward → next state):

```json
{"t": 0,   "obs": [...], "action": [...], "reward": 0.12, "terminated": false, "truncated": false, "info": {}}
{"t": 1,   "obs": [...], "action": [...], "reward": 0.15, "terminated": false, "truncated": false, "info": {}}
...
{"t": 998, "obs": [...], "action": [...], "reward": 0.18, "terminated": true,  "truncated": false, "info": {}}
```

**Field order** (fixed, for greppability):
`t / obs / action / reward / terminated / truncated / info / reward_components`

**Step semantics** follow Gymnasium 0.26+:
- `t` is the 0-based step index; `t ∈ [0, episode_length - 1]`.
- `obs` at step `t` is the observation **input to** the policy at that
  step (i.e., `obs_0` is from `reset()`; later `obs_t` are from
  `step()` of the previous transition).
- `action` is what the policy returned for `obs`.
- `reward` is the scalar reward received from the resulting transition.
- `terminated` = the episode ended naturally (agent died, goal
  reached, etc.). When `true`, future return = 0 (used by RL methods).
- `truncated` = the episode was cut off by a time limit
  (`max_episode_steps`) or external interrupt. When `true`, the
  underlying state may still be valid; future return ≠ 0 in general.
- An episode ends when either `terminated` or `truncated` is `true`
  (or both, in the rare case both happen at once). The next line will
  not exist within this `trajectory.jsonl`.

If `env_meta.reward_components` is non-empty (see §1.1), each step
additionally carries a `reward_components` field whose keys mirror
the declared component names:

```json
{"t": 0, "obs": [...], "action": [...], "reward": 0.12,
 "terminated": false, "truncated": false, "info": {...},
 "reward_components": {"forward": 0.15, "ctrl": -0.03, "contact": 0.0}}
```

The harness extracts each value from the `info` key declared in
`reward_components` (the raw value is also still present in `info`).
Per-step `reward_components` values sum to the per-step `reward` up
to the rounding the environment applies.

**Observation encoding** depends on `env_meta.obs_storage`:
- `"inline"` (default): `obs` is the observation JSON-serialized as
  nested lists. Atari RAM observations are 128-element integer lists;
  MuJoCo state observations are float lists. Each serialized `obs`
  MUST be ≤ 10 KB; envs whose `reset()` returns larger observations
  declare `obs_storage: "external"` (see §4.6).
- `"external"`: `obs` is `null`. Observations live in a side-car
  binary file `observations.npy` in the same `ep_<XXX>/` directory
  (see §4.6).

**Action encoding**:

The JSON form of each step's `action` field is determined entirely
by the env's declared `action_space` (see
`GET /info:env_meta.action_space`). Consumers MUST read
`action_space` before parsing `action` — the bare value is not
self-describing.

Standard Gymnasium action spaces serialize as follows:

| `action_space.type` | JSON form | Example |
|---|---|---|
| `Discrete` | int | `"action": 3` |
| `Box` | list of float (matching `shape`) | `"action": [0.1, -0.2]` |
| `MultiDiscrete` | list of int | `"action": [2, 0, 1]` |
| `MultiBinary` | list of 0/1 ints | `"action": [1, 0, 1, 1]` |
| `Dict` | JSON object keyed by sub-space names | `"action": {"throttle": 0.5, "gear": 2}` |
| `Tuple` | JSON list of sub-space values | `"action": [3, [0.1, 0.2]]` |

v1 supports only Gymnasium standard action spaces. Envs with
custom action types are out of scope; if a future env needs one,
the env registration MUST publish an explicit serialization recipe.

**Pre-clip vs. post-clip**: the recorded `action` value is **what
the policy returned**, before any env-side clipping or
transformation. If the env clips out-of-bounds actions (e.g., a
`Box` action returned as `[2.0, 0.0]` when bounds are `[-1, 1]`),
the recorded `action` is the original `[2.0, 0.0]`. The env MAY
record the post-clip value under `info["action_clipped"]` for the
agent's diagnostic use; this is env-defined and not required.

Rationale: replay tools want the exact value the policy produced;
the env's clipping is a deterministic function of the recorded
value plus `action_space`, recoverable when needed.

**NaN / Inf handling**: JSON does not support these literals.
Numeric fields containing NaN, +Inf, or -Inf are encoded as the
strings `"NaN"`, `"Inf"`, `"-Inf"` respectively. Consumers MUST
handle this when parsing numeric fields.

**Length**: the number of lines in `trajectory.jsonl` equals
`episode_lengths[i]` in the corresponding `summary.json`.

### 4.3 `episodes/ep_<XXX>/video.mp4` (lossy, human-facing)

Visualization-only video of the episode for **human** inspection.
**NOT a data source** — use `observations.npy` (if present) or
`trajectory.jsonl` for any programmatic obs access.

**Properties**:

| Property | Value |
|---|---|
| Container | MP4 |
| Codec | H.264 (any standard profile) |
| Frame rate | Environment's **native step rate** (1 video frame = 1 env step) |
| Resolution | Environment's native render resolution; harness does not scale |
| Color space | As returned by env's `render(mode="rgb_array")` |
| Quality | Lossy, targeting ~CRF 22 (human-readable, not bit-exact) |

**Frame–step synchronization** (critical):

> Video frame `N` shows the environment state at step `N` (i.e., the
> state the policy observed before its action at step `N` was applied).
> The total number of video frames equals `episode_lengths[i]` in
> `summary.json`, which equals the number of lines in
> `trajectory.jsonl`.

This invariant lets an agent locate a failure visually:
"summary.json says ep 042 errored at step 487" → seek to frame 487
in `video.mp4`.

**Existence conditions**:
- Present **only if** the env's `render(mode="rgb_array")` returns a
  non-null frame.
- For envs without any visual rendering (e.g., classic control with
  no render setup), `video.mp4` is **absent** — not an empty file.

**Failed episodes**: if an episode raises or times out at step K,
`video.mp4` contains frames 0..K (no padding, no extension). The
total frames equal the episode's actual length.

**Relationship with `observations.npy`**: when both files exist for
the same episode, they may differ in content:
- `observations.npy` is **lossless** and bit-exact to what the
  policy saw at each step.
- `video.mp4` is **lossy-compressed**; pixel values may shift by a
  few LSB from H.264 quantization.
- For some envs, the agent's observation and the human-facing video
  may use different camera angles or resolutions entirely (e.g.,
  agent sees a downsampled grayscale, video shows full-color
  third-person follow camera). The env declares both independently.

**Do not parse `video.mp4` for replay, training, or any analysis
where exact pixel values matter.** Lossy compression makes it
unsuitable for those purposes.

### 4.4 `errors.txt` (submit-level) and `episodes/ep_<XXX>/error.txt` (per-episode)

Despite the `.txt` extension (kept for human-readability via `cat`),
both error files are **JSON Lines**: one JSON object per line, each
recording a single error or timeout event.

The `.txt` extension signals "open me in any text editor and you can
read it"; the JSON Lines internal format signals "I'm also
machine-parseable line by line." Use `jq`, Python `json`, or any
JSONL tool to consume programmatically.

#### 4.4.1 Common event schema

Every entry, in both submit-level and per-episode files, has:

```json
{
  "schema_version": "0.1",
  "timestamp": "2026-05-28T10:01:23.456Z",
  "category": "init_error",
  "message": "Policy.__init__ raised: ValueError: invalid action_space dim",
  "traceback": "Traceback (most recent call last):\n  File \"system/policy.py\", line 14, ..."
}
```

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | Currently `"0.1"`. Required on every entry (each line is self-contained). |
| `timestamp` | ISO-8601 UTC string | `YYYY-MM-DDTHH:MM:SS.sssZ` |
| `category` | string | Categorizes the error; enum below |
| `message` | string | One-line human summary |
| `traceback` | string \| null | Full Python traceback for exceptions; `null` for non-exception events (e.g., timeouts) |

**`category` enum** (aligned with `summary.json:status` plus
episode-level categories):

| Value | Where used | Meaning |
|---|---|---|
| `oversize` | submit-level | Snapshot exceeded `system_total_bytes` |
| `missing_policy` | submit-level | No `policy.py` or no `Policy` class |
| `denied_import` | submit-level | Forbidden module imported |
| `import_error` | submit-level | Import raised (syntax, missing module, etc.) |
| `init_timeout` | submit-level | `Policy.__init__` exceeded `policy_load_wall_s` |
| `init_error` | submit-level | `Policy.__init__` raised |
| `invalid_env_instance` | submit-level | Requested env instance ID out of range `[0, n_env_instances)` |
| `oom` | submit-level | Process RSS exceeded `submit_peak_rss_bytes` during execution |
| `submit_wall_exceeded` | submit-level | Total wall time exceeded `submit_wall_s` during execution |
| `reset_error` | per-episode | `Policy.reset()` raised |
| `act_error` | per-episode | `Policy.act()` raised |
| `act_timeout` | per-episode | `act()` exceeded `act_wall_ms` |
| `on_episode_end_error` | per-episode | `Policy.on_episode_end()` raised |
| `truncated` | submit-level | Sentinel for additional events when error file is truncated (see §4.4.5) |

#### 4.4.2 Submit-level `errors.txt`

Present **only when `summary.json:status != "ok"`** — that is, the
submit failed before any episode ran. When present, no `episodes/`
directory is created. Typically has **exactly one entry** (the one
event that caused the failure):

```jsonl
{"schema_version":"0.1","timestamp":"2026-05-28T10:01:23.456Z","category":"denied_import","message":"Module 'transformers' is on the denied list (see AGENT.md §3.2)","traceback":null}
```

Or for an import error:

```jsonl
{"schema_version":"0.1","timestamp":"...","category":"import_error","message":"SyntaxError in system/policy.py","traceback":"Traceback ...\n  File \"system/policy.py\", line 5\n    def act(self obs):\n            ^\nSyntaxError: invalid syntax"}
```

#### 4.4.3 Per-episode `error.txt`

Present **only when this specific episode raised or timed out
mid-flight**. Each event during that episode becomes one line; in
practice there's usually one entry (the failure that ended the
episode), but multiple are allowed (e.g., a non-fatal warning
followed by the fatal error):

```jsonl
{"schema_version":"0.1","timestamp":"2026-05-28T10:02:15.789Z","category":"act_error","message":"act() raised at step 487","step_index":487,"traceback":"Traceback ...\n  File \"system/policy.py\", line 42, in act\n    return self.controller(obs)\n  File \"system/controllers/pid.py\", line 18, in __call__\n    raise ValueError(...)"}
```

The per-episode schema **adds one optional field** to the common
schema:

| Field | Type | Notes |
|---|---|---|
| `step_index` | int \| null | The step at which the event occurred (`null` for events outside `act()`, e.g., `reset_error`) |

For `act_timeout`, `message` says `"act() exceeded N ms wall time at step K"`, `traceback` is `null`:

```jsonl
{"schema_version":"0.1","timestamp":"...","category":"act_timeout","message":"act() exceeded 10ms wall time at step 312","step_index":312,"traceback":null}
```

The failing episode's `trajectory.jsonl` contains all steps up to
and including the failing step. For `act_error` or `act_timeout` at
step K, the action recorded at step K is the harness's fallback
(`action_space.sample()`; see §2.3).

#### 4.4.4 Mutual exclusion

The two locations are **mutually exclusive within a submit**:

- If submit-level `errors.txt` exists → `episodes/` directory does
  not exist (submit failed before any episode ran).
- If `episodes/` directory exists → no submit-level `errors.txt`
  (submit ran at least one episode); episode failures live in each
  `episodes/ep_<XXX>/error.txt`.

#### 4.4.5 Size cap

Each error file is capped at 64 KB. If multiple entries would exceed
this, later entries are truncated and a final line with
`category: "truncated"` and `message: "additional events omitted"` is
appended.

### 4.5 `episodes/ep_<XXX>/stdout.txt` and `episodes/ep_<XXX>/stderr.txt`

`XXX` is the global episode index (see §4.0). Capture the **policy's
own** standard output and standard error streams during this specific
episode (from the matching `reset()` through to `on_episode_end()`,
or to episode termination).

- `stdout.txt` collects everything written to `sys.stdout` (e.g.,
  `print("debug: x=", x)` from inside the policy).
- `stderr.txt` collects everything written to `sys.stderr` (including
  warnings raised via `warnings.warn`).

Both files are always created for each episode that runs. They are
empty (zero bytes) if the policy printed nothing during that episode.

Properties:

- **Per-episode scope**: each episode's output is isolated. To see
  output across all episodes in a submit, read each
  `ep_<XXX>/stdout.txt` in turn (e.g., `cat episodes/ep_*/stdout.txt`).
  No submit-level aggregation file is produced.
- **`__init__` output**: anything `Policy.__init__` prints (called
  once per submit, before any episode) goes into the first episode's
  `stdout.txt` (i.e., `ep_<first_global_episode>/stdout.txt`). If the
  submit fails
  before any episode runs, that output appears in submit-level
  `errors.txt` instead.
- **Size cap**: each file is truncated at 64 KB per episode. If
  truncation occurs, a final line `... [truncated at 64KB] ...` is
  appended.
- **Scope**: these capture **policy** output only. The harness's own
  logs, sandbox events, and env-side warnings do NOT appear here;
  they live in `runs/<...>/logs/` which is outside the agent's
  workspace (see `docs/output.md §6`).

Intended use: agent debugging. A common pattern is for the policy to
print diagnostic information (e.g., internal state, decision rationale)
on rare events; this content shows up in the corresponding episode's
`stdout.txt` for the next agent iteration to read.

### 4.6 `episodes/ep_<XXX>/observations.npy` (external obs only)

This file exists **only when `env_meta.obs_storage == "external"`**
(see §1.1). It holds the per-step observations as a binary array,
acting as the side-car to `trajectory.jsonl` whose `obs` fields are
then `null`.

**Why a side-car**: pixel-based environments (CarRacing, pixel Atari,
MuJoCo camera) produce observations too large to inline as JSON
nested lists. A 96×96×3 RGB frame is ~28 KB per step; 1000 steps
per episode × 8 episodes per submit easily exceeds 200 MB. JSON
encoding inflates this another 3–4×. Binary numpy storage is
compact, lossless, and supports random access via `mmap`.

**Format**:

| Property | Value |
|---|---|
| Shape | `[episode_length, *obs_shape]` |
| Dtype | matches the environment's observation space (e.g., `uint8` for pixel obs, `float32` for continuous) |
| Endianness | numpy native (`'<'` little-endian on the harness's machine; numpy persists this in the `.npy` header) |
| Ordering | row 0 is the observation at `t=0`, aligned with `trajectory.jsonl` line 0 |

**File extension**: the harness MAY write either `.npy` (uncompressed)
or `.npz` (compressed, single-array). Consumers MUST handle both:

```python
import numpy as np
from pathlib import Path

def load_obs(ep_dir: Path) -> np.ndarray:
    if (p := ep_dir / "observations.npy").exists():
        return np.load(p, mmap_mode='r')
    if (p := ep_dir / "observations.npz").exists():
        return np.load(p)["arr_0"]
    raise FileNotFoundError("no observations file")
```

For runs with many large episodes, `.npz` typically achieves 60–80%
size reduction on pixel data; the harness SHOULD prefer `.npz` for
pixel envs unless instructed otherwise.

**Random access for agent diagnosis**:

```python
import numpy as np
obs_arr = np.load("feedback/submit_005/episodes/ep_042/observations.npy",
                  mmap_mode='r')
frame_100 = obs_arr[100]   # shape == env's obs_shape
```

`mmap_mode='r'` avoids loading the entire array into RAM, so the
agent can inspect single frames without paying for the whole episode.

**Length consistency**: `observations.npy.shape[0]` MUST equal
`episode_lengths[i]` from `summary.json`, which equals the number of
lines in `trajectory.jsonl` for the same episode. The harness MUST
verify this on write.

**Failure semantics**: if an episode raises or times out mid-flight,
`observations.npy` contains observations up to and including the
failing step (same as `trajectory.jsonl`). The episode's `error.txt`
will be present.

**Video.mp4 relationship**: `observations.npy` and `video.mp4` are
**different artifacts** even when both exist. `observations.npy` is
**lossless** and what the policy actually saw; `video.mp4` is
**lossy-compressed** (H.264) for human inspection and may also use a
different camera angle or resolution than the agent's observation.
**Do not use `video.mp4` as a data source for replay or analysis**
where bit-exact observations matter.

### 4.7 Harness-level crashes (run abort)

A "harness crash" means the harness process itself dies mid-run for
reasons unrelated to the agent's policy: out-of-memory in the
harness, hardware failure, operator interrupt (Ctrl-C),
exhausted disk, etc. This is distinct from agent-side errors (which
are well-handled per §2.3 and §4.4).

When the harness crashes:

- `feedback/submit_NNN/` for the **in-progress** submit may be
  partially written or absent entirely. Consumers MUST tolerate
  missing or partial files inside `feedback/submit_<latest>/`.
- All previously **completed** submits' `feedback/submit_NNN/`
  directories are intact (they were written before the crash).
- The run's `run.json` (when generated by run resumption or
  cleanup) records `outcome.status = "error"` with `outcome.error`
  describing the crash.
- The harness MAY attempt cleanup writes; nothing is guaranteed.

The protocol does not promise per-submit atomicity in the face of
harness crashes. The run as a whole is the unit of atomicity:
either the run completes with a valid `run.json`, or the run is
marked errored and downstream tools should treat the entire run
as a failure.

This is a deliberate simplification: implementing per-submit
atomicity (e.g., write to `.partial/` then rename) would complicate
implementations across all four feedback files for marginal
benefit. Run-level failure is rare and easy to detect.

### 4.8 Feedback Structure Summary (reference card)

This subsection consolidates information distributed across §4.0–§4.7
into one navigable reference. Nothing new is introduced; if any
detail here conflicts with the per-file subsections above, those
subsections are normative.

#### 4.8.1 Overall map

```
workspace/feedback/
└── submit_NNN/                                              (one per submit)
    │
    ├── summary.json                                         ✅ ALWAYS
    │
    ├── errors.txt                                           ◯ XOR with episodes/
    │
    └── episodes/                                            ◯ XOR with errors.txt
        └── ep_<XXX>/                                        (one per attempted episode)
            ├── trajectory.jsonl                             ✅ always (may be empty)
            ├── stdout.txt                                   ✅ always (may be empty)
            ├── stderr.txt                                   ✅ always (may be empty)
            ├── observations.npy                             ◯ env_meta.obs_storage == "external"
            ├── video.mp4                                    ◯ env supports rendering
            └── error.txt                                    ◯ this episode failed mid-flight

Legend:  ✅ unconditionally created   ◯ conditional   XOR = mutually exclusive
```

#### 4.8.2 File existence matrix

| File | Created when | Notes |
|---|---|---|
| `submit_NNN/summary.json` | Every submit | Even on failure; `status` field tells which |
| `submit_NNN/errors.txt` | `summary.json:status != "ok"` | JSON Lines; describes submit-level failure (§4.4) |
| `submit_NNN/episodes/` (directory) | `summary.json:status == "ok"` | Contains one `ep_<XXX>/` per attempted episode |
| `episodes/ep_<XXX>/` (directory) | Each episode attempted (success or mid-flight failure) | Counter advances per attempt, see §4.0 |
| `episodes/ep_<XXX>/trajectory.jsonl` | Each `ep_<XXX>/` | Empty file if `reset()` raised before any step |
| `episodes/ep_<XXX>/stdout.txt` | Each `ep_<XXX>/` | Zero bytes if policy didn't print |
| `episodes/ep_<XXX>/stderr.txt` | Each `ep_<XXX>/` | Zero bytes if policy didn't print |
| `episodes/ep_<XXX>/observations.npy` | `env_meta.obs_storage == "external"` | May also be `.npz` (§4.6) |
| `episodes/ep_<XXX>/video.mp4` | Env's `render(mode="rgb_array")` returns non-null | Absent (not empty) if env can't render |
| `episodes/ep_<XXX>/error.txt` | Episode failed mid-flight (`reset_error`, `act_error`, `act_timeout`, `on_episode_end_error`) | JSON Lines; one or more entries (§4.4) |

#### 4.8.3 Mutual exclusion (per submit)

```
summary.json:status
        │
        ├── "ok"        ──→  episodes/ exists,  errors.txt does NOT exist
        │                    per-episode error.txt may exist for any failed episode
        │
        └── any non-"ok" ──→ errors.txt exists,  episodes/ does NOT exist
                             (no episodes ran)
```

The harness MUST NEVER create both `errors.txt` and `episodes/` for
the same submit. Validation tools (e.g., `hlbench check`) MUST verify
this invariant.

#### 4.8.4 Cross-file invariants

The following equalities and biconditionals hold and SHOULD be
verified by `hlbench check`.

| # | Invariant | Scope |
|---|---|---|
| F1 | `summary.json:status != "ok"` ⟺ `errors.txt` exists ⟺ `episodes/` does NOT exist | per submit |
| F2 | When `status == "ok"`: count of `episodes/ep_*/` directories == `summary.json:n_episodes` | per submit |
| F3 | When `status == "ok"`: directory names are `ep_<first_global_episode + i>/` for `i ∈ [0, n_episodes)` | per submit |
| F4 | `trajectory.jsonl` line count == `summary.json:episode_lengths[i]` | per episode (local index `i`) |
| F5 | If `observations.npy` exists: `observations.npy.shape[0]` == `trajectory.jsonl` line count | per episode |
| F6 | If `video.mp4` exists: frame count == `trajectory.jsonl` line count | per episode |
| F7 | `ep_<XXX>/error.txt` exists ⟺ local `i = XXX - first_global_episode` appears in `summary.json:errors` ∪ `summary.json:timeouts` | per episode |
| F8 | When `obs_storage == "external"`: every `obs` field in `trajectory.jsonl` is `null` | per episode |
| F9 | When `obs_storage == "inline"`: no `observations.npy` exists | per episode |

#### 4.8.5 Agent access patterns

Common things agents want to do and where to look:

| Goal | Path / query |
|---|---|
| Read the run's effective config | `GET /info` (static fields + state) |
| Read the current `remaining_budget` | `GET /info:state.remaining_budget` |
| Get the most recent submit's mean return | `submit_<latest>/summary.json:mean_return` |
| Plot the learning curve across submits | iterate `submit_*/summary.json:mean_return` (filter `status == "ok"`) |
| Find when policy first reached threshold X | iterate `summary.json` files until `mean_return ≥ X`, record `submit_index` |
| Inspect a specific episode's behavior | `episodes/ep_<XXX>/{trajectory.jsonl, video.mp4, stdout.txt}` |
| Find why a submit failed | check `summary.json:status`; if not `"ok"`, read `errors.txt` |
| Find which episode failed mid-flight | `summary.json:errors` and `summary.json:timeouts` → corresponding `ep_<XXX>/error.txt` |
| Get reward breakdown for diagnosis | `summary.json:reward_components_per_episode` |
| See what your policy printed at step 487 | locate the relevant `ep_<XXX>/stdout.txt`; print happens during step execution |
| Convert local episode index to global ID | `global = first_global_episode + local` |
| Convert global ID to (submit, local) | scan `summary.json` files; the one whose `[first_global_episode, first_global_episode + n_episodes)` contains the global ID |
| Estimate "thinking time" between submits | `submit_<N+1>:submit_started_at` − `submit_<N>:submit_completed_at` |
| Check whether held-out evaluation already ran | not visible from feedback; agent never sees held-out results |

#### 4.8.6 Implementer checklist

When implementing the harness's feedback writer, verify on every submit:

1. `summary.json` is always written, even on failure.
2. Exactly one of `errors.txt` / `episodes/` exists per submit, never both.
3. For each episode dir: `trajectory.jsonl`, `stdout.txt`, `stderr.txt` always created (may be empty).
4. `error.txt` is created **only** for failed episodes (do not create empty ones).
5. `observations.npy` (or `.npz`) is created **only** when `env_meta.obs_storage == "external"`.
6. `video.mp4` is created **only** when `env.render(mode="rgb_array")` returns a non-null frame.
7. All cross-file invariants F1–F9 hold.
8. Directory names use width `max(3, len(str(episode_budget)))`.

---

## 5. Scoring

### 5.1 Per-Submit Score

Reported in `summary.json.mean_return`. The mean undiscounted return
over the submit's episodes.

### 5.2 Final Score (Headline)

After the submit budget is exhausted (or the agent calls `finalize`):

1. Identify the agent's final policy: the most recent submit with
   `status == "ok"`. (Agents may explicitly designate an earlier
   submit; see §5.4.)
2. Run **all M held-out episodes** with hidden seeds loaded from the
   env's static `heldout.json` file (a separate static pool, disjoint
   from the in-loop env instances, completely invisible to the agent).
   M is determined by the env author at registration time (default
   256) and is **not exposed in `_run.json`** or anywhere else
   reachable by the agent.
3. Compute the **normalized score**:

   ```
   normalized = (mean_held_out_return - random_baseline) /
                (expert_baseline - random_baseline)
   score = clip(normalized, 0.0, 1.2) * 100
   ```

   - `random_baseline`: mean return of a uniformly random policy over
     the same M held-out episodes.
   - `expert_baseline`: published expert-level performance for the
     env. Declared in the env registration. **Server-internal: not
     exposed to the agent at any point** (not in `/info`, not in
     TASK.md). Agents work with raw `mean_return` values and
     optimize without knowing the target threshold.
   - The upper clip at 1.2 (120) allows super-expert performance to
     register but bounds outlier contributions.

Held-out evaluation results (mean, std, full return array) are
written to `run.json` outside the workspace. The agent never sees
held-out results, seeds, or trajectories at any point — during the
run or after it.

### 5.3 Auxiliary Metrics

Reported alongside the headline score, not part of it:

| Metric | Definition |
|---|---|
| `auc_in_loop` | Area under the in-loop `mean_return` curve plotted against cumulative episodes consumed, normalized to [0, 100] using the same baselines |
| `episodes_to_50pct` | Cumulative episodes consumed at the first submit whose `mean_return` exceeds 0.5 × expert; `null` if never |
| `episodes_to_80pct` | Same with 0.8 × expert |
| `held_out_gap` | `mean_in_loop_final - mean_held_out_final` (high positive = overfitting) |
| `n_submits` | Total submits made |
| `n_successful_submits` | Submits with `status == "ok"` |
| `mean_episodes_per_submit` | `episodes_used / n_submits` |
| `mean_submit_wall_time` | |

### 5.4 Explicit Final Submit Designation

By default, the final policy is the most recent successful submit.
An agent may override this by writing the desired submit index to
`system/.final_submit` (a single integer) before the budget is
exhausted. This is useful when the agent suspects regression in
recent submits.

---

## 6. Held-out Evaluation Details

### 6.1 Separate static pools

Each env defines two **static** seed pools, distributed as files
alongside the env code:

- **`train.json`** — array of N real seeds, one per env instance.
  Position `i` in the array maps to env instance ID `i` (the IDs the
  agent submits with `--env-instances`). `N = env_meta.n_env_instances`.
- **`heldout.json`** — array of M real seeds for held-out evaluation.
  Used only at run end, never accessible to the agent or its
  workspace. M is the env author's choice (default 256); not exposed
  to agents.

Both files are generated once by the env author (typically from a
single `master_seed` for reproducibility), committed to the env
package, and **frozen for that env version**. Updating the seeds
requires bumping `env_version`.

The agent sees:
- `n_env_instances` from `GET /info:env_meta.n_env_instances` → knows
  how many distinct env instances are addressable.
- Nothing about held-out (count, seeds, results) at any point.

### 6.2 Seed indirection

Agents address env instances by **integer ID** (0, 1, 2, ...), not
by real seed value. The label-to-seed mapping is internal to the
env's `train.json` and never exposed:

```
agent submit --env-instances 5
                    │
                    ▼
   harness reads train.json[5] → real_seed = (e.g.) 1097
                    │
                    ▼
   env initialized with real_seed=1097
```

Two consequences:
1. **Reproducibility**: env instance ID 5 always maps to the same
   real seed within a given `env_version`. Cross-run comparisons
   ("agent A vs agent B on env instance 5") are bit-exact.
2. **Abstraction**: agents never see real seed values. If a future
   `env_version` regenerates `train.json` with a different
   `master_seed`, env instance 5 maps to a different state — but
   agents always work with IDs, not seeds.

### 6.3 Determinism

Given an env instance ID, environment behavior is deterministic. The
policy's behavior is determined by its own code; if the policy uses
`numpy.random` or `random`, the harness seeds these with the env
instance's real seed at the start of `reset()`.

---

## 7. Run Reproducibility

A benchmark run is reproducible given:

- Env id and version (declared in the env registration; exposed via
  `GET /info:env` and `GET /info:env_version`, and in TASK.md).
- AGENT.md version (hash exposed via `GET /info:agent_md_hash`).
- The agent's submit-by-submit code snapshots (preserved by the harness).
- The harness version.

The workspace contains no run-config file; the authoritative source
of effective configuration is the server's `GET /info` endpoint
(schema in §1.1). The server holds this state in memory throughout
the run.

Final results (held-out score, auxiliary metrics, run summary) are
written to **`runs/<model>/<env>/<exp-id>/run.json`**, outside the
workspace. The agent has no access to it (eval happens after the agent
exits). See `docs/output.md §3` for the schema.

---

## 8. Out-of-Scope (For This Version)

The following are explicit non-features of v1 and may appear in
later versions:

- Multi-task workspaces / library reuse across tasks.
- Real-robot variants.
- Custom non-standard environments (envs not in the official registry).
- Distributed / parallel submit execution.

Pixel-input environments (CarRacing, MuJoCo camera, pixel Atari) **are
supported in v1** via the `obs_storage: "external"` mechanism
(see §1.1 and §4.6); they were initially scoped out but are now
in-scope.

Tasks added in later versions must remain compatible with this
specification.
