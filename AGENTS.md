# AGENTS.md — Rules of the Game

Authoritative rules for what agents may and may not do when solving
hlbench-pro tasks. Numbers and lists in this file are **defaults**;
the effective merged values for the current run are served by
`GET /info` — **`/info` always wins** if it disagrees with this doc.

This document is self-contained: everything you need to play the
game is here. Anything not in this file is not a rule.

---

## 1. The Core Invariant

> **All information your agent or its submitted policy uses about the
> environment must flow through the submit-feedback loop. The final
> policy must generalize to held-out evaluation episodes drawn from a
> disjoint, hidden seed pool.**

Everything else operationalizes this.

## 1.1 Your Goal

**Maximize `final_score` (§9).** Spend the entire episode budget
iterating; do not settle on the first policy that compiles or merely
produces non-zero return. The held-out evaluation rewards how high
you push policy quality, not how quickly you stop.

Concretely:
- Use the full `episode_budget` — every unspent episode is signal
  you didn't gather.
- After each submit, **diagnose** before re-submitting: read
  per-episode returns, trajectories, errors. Don't just guess.
- Try different ideas if the current direction plateaus. Re-using
  the same env_instance IDs across attempts lets you control
  variance and compare apples-to-apples.
- You're being judged on the policy's quality on **unseen seeds** —
  in-loop returns are a proxy, not the target.

---

## 2. Workspace Layout

```
workspace/
├── AGENTS.md        # this file (read-only)
├── system/          # your Python package — fully writable
│   └── policy.py    # required entry point (top level)
└── feedback/        # populated by server after each submit (read-only)
    └── submit_NNN/
        ├── summary.json
        └── episodes/ep_<XXX>/      # XXX = run-global episode index
            ├── trajectory.jsonl
            ├── stdout.txt          # captured policy stdout
            ├── stderr.txt
            ├── observations.npy    # only if env_meta.obs_storage == "external"
            ├── video.mp4           # only if env supports rendering
            └── error.txt           # only if this episode raised/timed out
```

The three top-level entries are the **only** things in the workspace.
Task description is fetched via **`GET /task`** (text/markdown), NOT
staged as a file. Effective config is fetched via **`GET /info`**.

Layout rules:
- `AGENTS.md` — global rules; read-only.
- `system/` — your code, persists across submits within a run; reset
  to empty starter at run start. Required: `system/policy.py` at top
  level. Free to add submodules / state files anywhere under `system/`.
- `feedback/` — append-only from your perspective. One `submit_NNN/`
  per submit, containing its `summary.json` and (on success) per-episode
  artifacts. Don't modify or delete.

What is **NOT** in your workspace and you have no read access to:
- Run-level outputs at `runs/<model>/<env>/<exp-id>/{run.json,
  checkpoints/, logs/}` — held-out results, harness logs, agent logs.
- Anything under `/tmp`, `~/`, or anywhere outside `workspace/`.

If you need diagnostic info, `print()` it from inside `policy.py` —
it lands in `feedback/submit_NNN/episodes/ep_<XXX>/stdout.txt` for
the next iteration.

To carry state across submits (since the `Policy` instance is
destroyed after each submit), write to files under `system/` from
inside `act()` / `reset()` and read them in `__init__`.

---

## 3. Your Policy

`system/policy.py` MUST define a class named `Policy` with this
exact contract:

```python
import numpy as np
from typing import Any, Mapping

class Policy:
    def __init__(
        self,
        obs_space: Mapping[str, Any],
        action_space: Mapping[str, Any],
        env_meta: Mapping[str, Any],
    ) -> None:
        """Constructed once per submit (shared across all episodes
        of that submit). May read files from system/. Must not read
        from feedback/ or anywhere outside workspace/.

        env_meta carries the same fields as GET /info:env_meta plus:
            env: str                       (env slug)
            submit_index: int              (0-based; increments per submit)
            n_episodes_this_submit: int
            remaining_budget_after: int
            max_episode_steps: int
            allowed_imports: tuple[str, ...] (informational)
        """
        ...

    def reset(self, episode_index: int) -> None:
        """Called at the start of every episode in the submit.
        episode_index ranges over [0, n_episodes_this_submit).
        The episode seed is NOT passed and MUST NOT be inferred."""
        ...

    def act(self, obs) -> Any:
        """Called once per step. Must return an action valid for
        action_space. Wall-time limit per call (see §4.3)."""
        ...
```

There is intentionally **no** `on_episode_end` hook — episode
returns reach you through `feedback/submit_NNN/summary.json:returns`
after the submit completes.

`obs` and `action` shapes / dtypes / ranges follow Gymnasium
conventions and are documented in `GET /info:env_meta.obs_space` /
`env_meta.action_space`. Standard space encodings:

| `space.type` | obs/action JSON form | typical Python type |
|---|---|---|
| `Box` | flat list of floats matching `shape` | `numpy.ndarray` |
| `Discrete` | int | `int` |
| `MultiDiscrete` | list of ints | `numpy.ndarray` of int |
| `MultiBinary` | list of 0/1 ints | `numpy.ndarray` of int |
| `Dict` | JSON object keyed by sub-space names | `dict[str, ...]` |
| `Tuple` | JSON list of sub-space values | `tuple[...]` |

Per-step values in `trajectory.jsonl` use these forms. NaN / Inf are
encoded as the strings `"NaN"`, `"Inf"`, `"-Inf"`.

### 3.1 Errors

| Error | When | Consequence |
|---|---|---|
| `Policy.__init__` raises | submit-level, before any episode | `summary.json:status = "init_error"`; full requested `N` consumed |
| `Policy.__init__` exceeds wall time | same | `init_timeout`; full `N` consumed |
| `Policy.reset()` raises | per episode | episode counted, empty `trajectory.jsonl`, `error.txt:category = "reset_error"`; subsequent episodes proceed with fresh `reset()` |
| `act()` raises mid-episode | per episode | episode terminates with return so far; failing step's action is `action_space.sample()`; `error.txt:category = "act_error"`; later episodes continue |
| `act()` exceeds wall time | per episode | action defaults to `action_space.sample()`, episode continues; `error.txt:category = "act_timeout"` per occurrence |

### 3.2 Python path

When `policy.py` is loaded, `sys.path[0] = workspace/system/` and
`os.getcwd() = workspace/system/`. Imports between your modules
resolve naturally:

```python
# In system/policy.py
from controllers.main import Controller
from utils.helpers import preprocess
import json
```

Both regular packages (with `__init__.py`) and namespace packages
(without) work. Don't manipulate `sys.path` at runtime to escape
the workspace — that violates §6.

---

## 4. Sandbox Rules (Hard, Enforced)

### 4.1 Network

- **No network access** while `policy.py` executes.
- Network is allowed **between submits** (while you reason / edit
  code); every outbound request is logged and included in the run
  report.

### 4.2 Imports

Import policy is enforced by a `sys.meta_path` hook on `policy.py`
and any module it imports. **Forbidden imports raise `ImportError`
at submit time — the submit consumes its full committed budget and
earns the `denied_import` verdict.**

| Allowed | Forbidden | Why forbidden |
|---|---|---|
| `numpy`, `scipy` (except `scipy.optimize.{minimize, differential_evolution, basinhopping, dual_annealing}`) | `transformers`, `huggingface_hub`, `timm`, `diffusers` | trivially load pretrained weights → violates §1 |
| `math`, `collections`, `itertools`, `functools`, `dataclasses`, `typing`, `enum`, `heapq`, `bisect`, `re`, `json`, `pickle`, `pathlib`, `time`, `random` | `openai`, `anthropic`, `google.genai`, `cohere` | external model API → violates §1 |
| `torch`, `jax`, `flax` | `stable_baselines3`, `ray`, `rllib`, `cleanrl`, `sb3_contrib` | bundle pretrained checkpoints / bypass submit interface |
| | `urllib`, `requests`, `socket`, `httpx`, `aiohttp` | enable network → violates §4.1 |
| | `subprocess`, `os.system`, `os.exec*` | sandbox escape vectors |

The full enforced list is at `GET /info:denied_imports`.

### 4.3 Resource Limits

Defaults below; effective values at `GET /info:resource_limits`.

| Limit | Default | Notes |
|---|---|---|
| `act()` wall time | 10 ms | per call |
| `Policy.__init__` wall time | 1 s | from import to first `reset()` |
| Per-submit wall time | 5 min | total across all episodes in the submit |
| Per-submit peak RSS | 1 GB | |

### 4.4 Persistence

- `system/` persists across submits within a run.
- `system/` is reset to starter contents at run start.
- Free to delete, modify, restructure under `system/` — only
  constraint is that `policy.py` must always exist and define a
  valid `Policy` class (§3).

---

## 5. Submit Protocol

### 5.1 The four endpoints

The per-run server runs on the same host as you. URL discovery, in
priority order:

1. environment variable `HLBENCH_SERVER_URL` (e.g. `http://127.0.0.1:54321`)
2. file `workspace/.server_url` (single line containing the URL)

| Endpoint | Method | Purpose |
|---|---|---|
| `/info` | GET | Static config + dynamic state (call at start; refresh between submits) |
| `/task` | GET | Human-readable task description (text/markdown) |
| `/submit` | POST | Run episodes on chosen env instances (sync; blocks) |
| `/finalize` | POST | Declare run finished. In automated harnesses you typically don't call this — the harness handles it once budget is spent |

`POST /submit` body: `{"env_instances": [int, ...]}`. Response:
`{"submit_id", "status", "summary"}` — `summary` is byte-identical
to `workspace/feedback/submit_<id>/summary.json` which the server
writes before returning. Per-episode artifacts (trajectory, video,
stdout/stderr, error) are read directly from disk after `/submit`
returns.

### 5.2 Budget rules

The benchmark allocates **a single resource: total episodes**. Each
submit consumes `len(env_instances)` from the budget.

| Parameter | Default | Where to read live value |
|---|---|---|
| `episode_budget` | 256 | `GET /info:episode_budget` |
| `min_episodes_per_submit` | 1 | `GET /info:min_episodes_per_submit` |
| `max_episodes_per_submit` | 256 | `GET /info:max_episodes_per_submit` |
| `n_env_instances` | env-defined (e.g. 10000) | `GET /info:env_meta.n_env_instances` |
| `remaining_budget` | live | `GET /info:state.remaining_budget` |

You address an env instance by integer ID in `[0, n_env_instances)`.
The mapping ID → real seed is server-internal and never exposed.
Re-submitting the same env_instance ID gives the same trajectory
under a deterministic policy (useful for variance estimation or
regression testing). Each ID submitted consumes one episode, even
the duplicates.

**Held-out evaluation** — size, seeds, individual returns, and
expert/random baselines are **never exposed**. You optimize against
raw `mean_return` without knowing what target threshold means
"expert level".

### 5.3 Submit lifecycle and verdicts

Each submit progresses through phases internally
(**Request → Snapshot → Validate → Compile → Initialize → Execute →
Commit**) and emits exactly one **verdict** in `summary.json:status`.

| Verdict | Meaning | Budget consumed? |
|---|---|---|
| `ok` | All requested episodes ran | yes (`N` requested) |
| `budget_invalid` | requested count outside `[min, max, remaining]` | **no** (free retry) |
| `invalid_env_instance` | requested ID outside `[0, n_env_instances)` | **no** (free retry) |
| `missing_policy` | no `policy.py` or no `Policy` class | yes (`N`) |
| `denied_import` | snapshot imported a forbidden module | yes (`N`) |
| `import_error` | snapshot import raised (syntax / missing module / etc.) | yes (`N`) |
| `init_timeout` | `Policy.__init__` exceeded wall time | yes (`N`) |
| `init_error` | `Policy.__init__` raised | yes (`N`) |
| `oom` | RSS exceeded `submit_peak_rss_bytes` mid-execute | yes (`N`); partial episodes preserved |
| `submit_wall_exceeded` | wall time exceeded mid-execute | yes (`N`); partial episodes preserved |

**Once snapshot is taken (i.e. anything past the request validation
step), the full requested `N` is committed regardless of outcome.**
The only "free rejection" is malformed parameters
(`budget_invalid` / `invalid_env_instance`).

Per-episode failures inside a successful submit (`reset_error`,
`act_error`, `act_timeout`) live in
`episodes/ep_<XXX>/error.txt` — they don't change the
submit-level verdict. See §3.1.

### 5.4 Cadence

You decide how to spend the budget. There's no fixed cadence.
Reference patterns:

| Pattern | Episodes per submit | When |
|---|---|---|
| Cheap probe | 1–3 | Sanity check after a code change |
| Standard | 4–8 | Default; reasonable signal/cost |
| High-confidence | 16–32 | Comparing two candidates |
| Burn budget | rest | Last submit |

Strategic budget allocation is itself a tested capability.

### 5.5 End of run

When `remaining_budget == 0` (or in automated harnesses, when the
harness decides to finalize), the **most recent successful submit**
(`status == "ok"`) is used as the final policy for held-out
evaluation. There is no agent-side override mechanism — to "go
back" to an earlier policy you must keep your own backup under
`system/` and re-submit it.

Held-out runs once. You don't see its seeds, parameters, or
per-episode results.

---

## 6. Anti-Hack Rules

These exist solely to enforce §1. They are not methodological
preferences. **Violation disqualifies the run.**

1. **No pretrained model loading.** No model weights, parameters,
   or data files not produced by your own code in `system/` during
   the current run.
2. **No held-out seed access.** Eval seeds are not in env vars,
   files, or any other channel. Don't try to enumerate or guess
   them.
3. **No external compute during policy execution.** `policy.py`
   runs entirely inside the sandbox. No process spawning, sockets,
   or external services.
4. **No reading outside the workspace.** `policy.py` may only read
   files under `workspace/system/`.

---

## 7. Method Neutrality

hlbench-pro does not prescribe how to solve a task. **Any approach
that respects §4 (sandbox) and §6 (anti-hack) is fair game** — the
benchmark scores the resulting policy, not the method used to produce
it.

The rollout budget is intentionally tight. How to spend it
efficiently is itself part of what's being tested.

---

## 8. Per-Run Configuration

The effective config is the merge of three layers, by increasing
precedence:

1. **Global defaults** (this document — §4 sandbox, §5 budget).
2. **Env declarations** (registered envs may override
   `episode_budget`, `n_env_instances`, `max_episode_steps`,
   `expert_baseline`, `reward_components`, recommended limits).
3. **Run-time CLI overrides** (e.g. `--budget 32`).

Later layers may **tighten** earlier ones (smaller budget, stricter
limits, additional forbidden imports). Later layers MAY NOT relax
the anti-hack rules in §6 or grant network access.

The merged result is at **`GET /info`** — that endpoint is the only
authoritative source. Always call it at run start.

---

## 9. Scoring

Held-out evaluation runs all M held-out episodes through your final
policy. The headline score is

```
final_score = clip((mean_held_out − random_baseline) /
                   (expert_baseline − random_baseline), 0, 1.2) × 100
```

`random_baseline` and `expert_baseline` are server-internal — you
never see their values. The clip at 1.2 (= 120) lets super-expert
performance register but bounds outliers.

You will only ever see the `final_score` (and only after the run
has ended, via the harness's reporting channel). Held-out
per-episode returns and aggregates remain hidden from the agent
at every point — during the run and after.

Auxiliary metrics computed at finalize time (`auc_in_loop`,
`episodes_to_50pct`, `episodes_to_80pct`, `held_out_gap`,
`n_submits`, `episodes_used`, `mean_episodes_per_submit`,
`mean_submit_wall_time`) are reported alongside the headline but
not shown to the agent.
