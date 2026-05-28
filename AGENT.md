# AGENT.md — Rules of the Game

This document defines what agents may and may not do when solving
hlbench-pro tasks. These rules apply globally to all tasks. Individual
envs MAY tighten (but not relax) these rules via their env
registration; run-time invocations MAY further tighten via CLI flags
or config files. The effective merged values for the current run are
served by the per-run server via `GET /info` — that endpoint is
authoritative.

If anything in `GET /info` conflicts with the defaults shown in this
document, `/info` wins.

---

## 1. The Core Invariant

> **All information about the environment used by your agent or its
> submitted policy must flow through the submit-feedback loop. The
> final submitted policy must generalize to held-out evaluation
> episodes drawn from a disjoint, hidden seed pool.**

This is the single principle that defines a valid solution.
Everything below operationalizes it.

---

## 2. Workspace Layout

You operate inside a per-task workspace with this layout:

```
workspace/
├── TASK.md          # task description (delivered by server at run start, read-only)
├── AGENT.md         # this file (read-only)
├── system/          # your Python package — fully writable
│   ├── policy.py    # required entry point (top level, defines Policy class)
│   ├── controllers/ # example: organize code as packages or flat files
│   ├── utils/       # example: helper modules
│   ├── memory/      # example: persistent state files (.json, .npy)
│   └── tests/       # example: self-tests you run yourself
└── feedback/        # populated by server directly into shared workspace (read-only)
    └── submit_000/
        ├── summary.json
        ├── episodes/
        │   └── ep_<XXX>/       # XXX = run-global episode index, zero-padded
        │       ├── trajectory.jsonl
        │       ├── video.mp4
        │       ├── observations.npy   # only if env declares external obs storage
        │       ├── stdout.txt
        │       ├── stderr.txt
        │       └── error.txt          # only if this episode raised/timed out
        └── errors.txt                  # only if submit-level failure (no episodes ran)
```

You may organize `system/` as a flat collection of files or as a
nested Python package (with or without `__init__.py`). When
`policy.py` is loaded, the harness places `system/` at `sys.path[0]`,
so imports like `from controllers.pid import PIDController` resolve
naturally. See `SPEC.md §2.4` for the full import semantics.

Common patterns: helper modules, controller libraries, persistent
state files, ablation snapshots, self-tests. `system/` persists
across submits within a run.

Effective configuration for the current run (episode budget, resource
limits, allowed imports, env metadata, dynamic state like
`remaining_budget`) is served by the per-run server via `GET /info`.
Call `/info` at run start to read static config and `env_meta`; call
again after each submit (or periodically) to refresh dynamic state.
The numbers shown elsewhere in this document are defaults and may be
overridden by env registration or run-time CLI flags.

`feedback/` is append-only from your perspective: each submit produces
a new `submit_NNN/` directory. You may read but not modify or delete it.

### What is NOT in your workspace

The harness also writes run-level artifacts (final score, harness
logs, agent harness logs, env logs, code checkpoint archive) to a
location **outside your workspace** (e.g., `runs/<model>/<env>/<exp-id>/`).
These are for post-hoc analysis by humans or analysis tools; **you
have no read access to them**. Specifically:

- You cannot read `runs/<...>/logs/harness.log`, `agent.jsonl`, or
  `env.log`. Any diagnostic information you need to iterate on your
  policy is delivered to you via `feedback/`.
- You cannot read the final held-out evaluation results — those are
  computed after your run terminates.

If you find yourself needing information that is not in `feedback/`,
the answer is to print it from inside `policy.py` (it will appear in
`feedback/submit_NNN/episodes/ep_<XXX>/stdout.txt` for the episode in
which it was printed), not to look elsewhere.

---

## 3. Sandbox Rules (Hard, Enforced)

### 3.1 Network
- **No network access** while `policy.py` is executing.
- Network access is permitted while the agent is reasoning or editing
  code (between submits), but every outbound request is logged and
  included in the run report.

### 3.2 Imports

Allowed inside `policy.py` and any `system/*.py` module it imports:

| Category | Modules |
|---|---|
| Numerical | `numpy`, `scipy` (excluding `scipy.optimize.{minimize, differential_evolution, basinhopping, dual_annealing}`) |
| Standard library | `math`, `collections`, `itertools`, `functools`, `dataclasses`, `typing`, `enum`, `heapq`, `bisect`, `re`, `json`, `pickle`, `pathlib`, `time`, `random` |
| ML frameworks | `torch`, `jax`, `flax` |

Forbidden:

| Category | Why |
|---|---|
| `transformers`, `huggingface_hub`, `timm`, `diffusers` | Trivially load pretrained weights — violates invariant |
| `openai`, `anthropic`, `google.genai`, `cohere` | External model API — violates invariant |
| `stable_baselines3`, `ray`, `rllib`, `cleanrl`, `sb3_contrib` | Bundle pretrained checkpoints and bypass the submit interface |
| `urllib`, `requests`, `socket`, `httpx`, `aiohttp` | Enable network access — violates 3.1 |
| `subprocess`, `os.system`, `os.exec*` | Sandbox escape vectors |

The allow/deny lists are enforced by an import hook. Attempts to
import a forbidden module raise an `ImportError` at submit time and
the submit counts as a failed submit (counts toward your budget,
returns zero reward).

### 3.3 Resource Limits

The following limits are enforced by the sandbox. Default values are
shown below; **the effective values for the current run are in
`GET /info:resource_limits`** and may differ.

| Limit | Default | Notes |
|---|---|---|
| `system/` total size | 50 KB | Source files only (see size rule below) |
| Single file size in `system/` | 25 KB | Prevents a single large weight blob |
| `act()` wall time | 10 ms | Per call |
| Policy load time (`__init__`) | 1 s | From import to first `reset()` |
| Per-submit wall time | 5 min | Total across all episodes in a submit |
| Per-submit peak RSS | 1 GB | |

Limits may be raised per task when justified (e.g., planning-heavy
tasks may need 100 ms `act()`). Effective values are always in
`GET /info:resource_limits`.

**`system/` size calculation rule.** The harness sums the sizes of
all files under `system/` that match the following criteria:

- **Counted**: any file the agent created or modified — `.py`,
  `.json`, `.yaml`, `.yml`, `.npy`, `.npz`, `.csv`, `.txt`, `.md`,
  or any other extension. Hidden files (starting with `.`) are
  counted too (including the agent's optional `.final_submit`).
- **Not counted (auto-excluded by the harness)**:
  - `__pycache__/` and `*.pyc` (Python bytecode cache)
  - `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` (tool caches)
  - `.git/` (in case agent ran git locally)
  - Symbolic links (forbidden anyway)

Rationale: the limit governs the agent's deliberate code and data,
not transient artifacts that any Python process generates.

### 3.4 Persistence

- `system/` persists across submits within a single benchmark run.
- `system/` is reset to its starter contents at the beginning of each run.
- You may delete, modify, or restructure anything under `system/`,
  except for the requirement that `system/policy.py` always exists
  and defines a valid `Policy` class (see `SPEC.md §2`).

---

## 4. Submit Protocol

The benchmark allocates a single resource: the **total episode budget**.
Each submit consumes a number of episodes specified by the agent at
submit time. When the budget reaches zero, no more submits are allowed.

### 4.0 HTTP Interface

You communicate with the per-run server **via HTTP** (the server runs
on the same host as you). Three endpoints cover the entire control
surface:

| Endpoint | Method | Purpose |
|---|---|---|
| `/info` | GET | Read static config + dynamic state |
| `/submit` | POST | Run episodes on selected env instances |
| `/finalize` | POST | Declare run complete, trigger held-out eval |

The server URL is provided to the agent harness at startup via:
- environment variable `HLBENCH_SERVER_URL` (e.g., `http://localhost:7000`), or
- file `workspace/.server_url` (a single line containing the URL).

**Submit is synchronous**: `POST /submit` blocks until the server has
finished running the requested episodes and writing all feedback
files to `workspace/feedback/submit_NNN/`. The HTTP response carries
the same content as `summary.json`, so you can act on the result
immediately without re-reading the file.

`GET /info` example:

```http
GET /info HTTP/1.1
```

```json
{
  "schema_version": "0.1",
  "env": "halfcheetah",
  "env_version": "0.1",
  "episode_budget": 256,
  "min_episodes_per_submit": 1,
  "max_episodes_per_submit": 256,
  "resource_limits": { ... },
  "allowed_imports": [...],
  "denied_imports": [...],
  "env_meta": {
    "obs_space": {...}, "action_space": {...},
    "max_episode_steps": 1000,
    "n_env_instances": 256,
    "obs_storage": "inline",
    "reward_components": { ... }
  },
  "state": {
    "remaining_budget": 248,
    "n_submits": 3,
    "last_submit_status": "ok",
    "is_finalized": false
  }
}
```

`POST /submit` example:

```http
POST /submit HTTP/1.1
Content-Type: application/json

{ "env_instances": [0, 1, 2, 3, 4, 5, 6, 7] }
```

```json
{
  "submit_id": 5,
  "status": "ok",
  "summary": { /* same as workspace/feedback/submit_005/summary.json */ }
}
```

For per-episode artifacts (trajectory, video, observations, stdout,
stderr, error files), read them directly from
`workspace/feedback/submit_<id>/episodes/ep_<XXX>/`. The server has
already written them by the time `/submit` returns.

`POST /finalize` example:

```http
POST /finalize HTTP/1.1
```

```json
{
  "status": "evaluating"
}
```

`/finalize` triggers held-out evaluation; results are written to
`runs/<...>/run.json` outside your workspace. You will not see them.

### 4.1 Budget

| Parameter | Where to find effective value | Notes |
|---|---|---|
| Episode budget | `GET /info:episode_budget` | Default 256; env registration may override; CLI may override |
| Min episodes per submit | `GET /info:min_episodes_per_submit` | Default 1 |
| Max episodes per submit | `GET /info:max_episodes_per_submit` | Default = remaining budget |
| Env instance count | `GET /info:env_meta.n_env_instances` | Number of distinct env instances available (agent submits by ID `[0, N)`); default 256, env-overridable |
| Remaining budget (live) | `GET /info:state.remaining_budget` | Refreshes on every call |
| Held-out evaluation | **Hidden** | Size, seeds, results: none exposed |

The agent chooses **which env instances** each submit runs by passing
their integer IDs (e.g., `--env-instances 0-19`). Each ID run consumes
one episode from the budget. Re-submitting the same ID multiple times
is allowed (useful for variance estimation or regression testing
after code changes) and each run consumes one episode.

- **Cheap probe** (1–3 episodes): quickly check whether a code change
  shows any signal. High variance per submit, low cost.
- **Standard evaluation** (8 episodes, the conventional default):
  reasonable signal-to-noise for most dense-reward tasks.
- **High-confidence evaluation** (16–32 episodes): low-variance
  comparison between candidate policies, useful late in a run.

Strategic budget allocation is itself a tested capability. There is
no fixed cadence imposed by the harness.

### 4.2 What a submit does

1. The harness reads the requested env instance IDs from the agent's
   submit invocation (e.g., `--env-instances 0-19`), expands the spec
   to a concrete list, and validates:
   - every ID is in `[0, n_env_instances)` (else verdict
     `invalid_env_instance`, no budget consumed);
   - the count satisfies budget bounds (else `budget_invalid`).
2. Snapshots `system/` at submit time.
3. Validates the snapshot: size limit, import scan, `Policy` class
   present and importable.
4. If validation fails, writes `feedback/submit_NNN/errors.txt` and
   `summary.json` with the appropriate non-`ok` status (`oversize`,
   `missing_policy`, `denied_import`, `import_error`, `init_timeout`,
   or `init_error`; see SPEC.md §4.1 for the full enum). **The full
   committed episode count is still consumed from the budget**, since
   the agent committed those resources at submit time.
5. Otherwise, instantiates `Policy` and runs one episode per requested
   env instance ID (each using its associated hidden real seed loaded
   from the env's `train.json`).
6. Writes `feedback/submit_NNN/` with rewards, trajectories, optional
   videos, and any errors raised during the run.
7. Decrements `remaining_budget` by the number of env instances run.

`remaining_budget` after each submit is reported in
`feedback/submit_NNN/summary.json`.

### 4.3 End of run

When `remaining_budget == 0`, or when the agent declares the run
finished, the most recent successful submit is used as the final
policy for held-out evaluation (unless overridden via
`system/.final_submit`; see SPEC.md §5.4).

Held-out evaluation is run by the harness once. The agent does not
see its seeds, parameters, or per-episode results — only the final
score after the run ends.

---

## 5. Anti-Hack Rules

These rules exist solely to enforce the Core Invariant. They are not
methodological preferences.

1. **No pretrained model loading.** You may not load model weights,
   parameters, or any data file not produced by your own code in
   `system/` during the current run.
2. **No held-out seed access.** Eval seeds are not exposed via
   environment variables, files, or any other channel. Do not attempt
   to enumerate or guess them.
3. **No external compute during policy execution.** `policy.py` runs
   entirely inside the sandbox; spawning processes, opening sockets,
   or calling external services is forbidden.
4. **No reading outside the workspace.** `policy.py` may read files
   only under `workspace/system/`. Reads from `workspace/feedback/`,
   `/tmp`, the home directory, or anywhere else raise an error.

Violation of any anti-hack rule disqualifies the run.

---

## 6. Method Neutrality (Informational)

hlbench-pro does not prescribe how you solve a task. Permitted
approaches include but are not limited to:

- Rule-based controllers, state machines, behavior trees
- Classical control (PID, LQR, MPC) with hand-derived parameters
- Search and planning (MCTS, A*, sampling-based MPC)
- Neural networks trained from scratch using rollout data
- Hybrid policies combining any of the above

The rollout budget (default 30 submits × 8 episodes) is intentionally
tight. Approaches that require large amounts of environment interaction
to perform well — such as training PPO or SAC from scratch — will
typically underperform approaches that extract more information per
episode. **This is an outcome of the budget, not a prohibition.**

---

## 7. Per-Run Configuration

The effective configuration for the current run is the result of
merging three layers, in increasing precedence:

1. **Global defaults** defined in this document (§3 sandbox, §4
   budget defaults).
2. **Env declarations** from the environment registration code (e.g.,
   the halfcheetah env declares its default `episode_budget`,
   `n_env_instances`, `max_episode_steps`, `expert_baseline`,
   `reward_components`, recommended resource limits).
3. **Run-time overrides** passed to `hlbench run` via CLI flags
   (e.g., `--episode-budget 500`) or a config file.

Later layers may **tighten** earlier ones (smaller budget, stricter
resource limit, additional forbidden imports). Later layers MAY NOT
relax the anti-hack rules in §5 or grant network access.

Configurable parameters (any of which can come from layer 2 or 3):

- `episode_budget` (integer)
- `n_env_instances` (integer; env-level, in env_meta)
- `min_episodes_per_submit`, `max_episodes_per_submit` (integers)
- Resource limits (any of §3.3)
- Imports (additional allow or deny entries)

The merged result is served by the per-run server via `GET /info`
and a prose summary may be reflected in `TASK.md`. **Always call
`GET /info` at the start of work** — it is the authoritative
effective config. The numbers in this document are defaults that
may have been overridden. The agent does not see which layer
contributed which value; only the merged result.

---

## 8. What is Scored

Held-out final return, normalized against random and expert baselines
defined per task. See `SPEC.md §5` for the scoring formula and
auxiliary metrics.

The agent is never shown held-out results during the run. Only the
final held-out score is reported, after the submit budget is consumed.
