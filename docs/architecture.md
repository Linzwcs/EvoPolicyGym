# docs/architecture.md — MVP Implementation Plan (Pendulum Pilot)

This document is the **implementation plan for the v0 MVP**, focused
on getting a single env (Pendulum-v1) end-to-end. It is not a
complete architecture for all of hlbench — it is the smallest thing
we can build that exercises the protocol and gives us real data.

> **Status (Day 14):** the plan executed. All 14 days are complete;
> 77 tests pass; reference PD on Pendulum hits `final_score = 98.3`.
> Section 3 below shows the *as-built* layout (with deviations from
> the original sketch noted inline). Sections 4 and 5 are kept as
> the historical design + build sequence — useful when extending
> past Pendulum, less useful as "what's in the code". For the
> current code map, prefer `README.md` and `docs/quickstart.md`.

For the full protocol contract, see `SPEC.md`, `AGENTS.md`,
`docs/output.md`, `docs/submit-protocol.md`.

---

## 1. Goal

**In ~2 weeks**, have a working system that:

1. Loads the Pendulum-v1 environment via Gymnasium.
2. Accepts a `Policy` class from agent code.
3. Runs N submits, each with M env_instances → produces feedback files.
4. Triggers held-out evaluation at run end.
5. Writes a `run.json` with final score.

By the end, we should be able to write a 30-line PD controller, run
`hlbench submit --env-instances 0-7`, see feedback, iterate, and get
a normalized final score against a hidden held-out pool.

## 2. Scope

### In scope (v0 MVP)

| Component | MVP form |
|---|---|
| Env | **Pendulum-v1 only** |
| Agent harness | A reference Python script (hand-written PD controller) using `httpx` to call server |
| Server | Python lib (`hlbench.core.Server`) wrapped in **minimal FastAPI** (3 endpoints) |
| Transport (agent ↔ server) | **HTTP** — `GET /info`, `POST /submit`, `POST /finalize` |
| Transport (feedback) | **Shared filesystem** — server writes `workspace/feedback/` directly; agent reads as files |
| Submit | **Synchronous HTTP** (POST blocks until server finishes) |
| Sandbox | Subprocess with `signal.alarm` for `act()` timeout; basic `rlimit` for memory |
| Held-out | Static `heldout.json` array, run sync at finalize |
| Persistence | Filesystem only (no DB) |
| CLI | Minimal: `init`, `submit`, `info`, `finalize` (wraps HTTP calls) |

### Out of scope (defer)

| Deferred | Why |
|---|---|
| Async + long-poll | Local HTTP, no timeout pressure even for long episodes |
| Tar bundle transport | Shared filesystem makes it unnecessary |
| Remote / distributed deployment | Shared-host is the **permanent** architecture |
| Container / Docker | Local subprocess works for MVP |
| Pixel observations | Pendulum is state-based (3 floats) |
| Multi-env | Pendulum first, others added after |
| Agent.jsonl logging | Useful but not critical for MVP |
| `checkpoints/` directory | Skip for v0; add when needed |
| `denied_imports` enforcement | Pendulum PD controller uses only numpy — skip filter for now |
| Network blocking | Same: PD controller doesn't network — skip |
| Run resumption | If server crashes, start fresh |

These deferred items are **real spec items** to add later — but the
MVP doesn't need them to validate the core loop.

## 3. Package Layout

Below is the **as-built** layout (what's on disk today). The design
sketch matched closely; deviations from the original sketch are
documented inline.

```
hlbench-pro/
├── src/hlbench/                # SERVER LIBRARY (no consumers below this line)
│   ├── __init__.py             # __version__
│   ├── http_server.py          # stdlib http.server wrapper around Server
│   ├── core/
│   │   ├── __init__.py
│   │   ├── server.py           # Server class (entry point, init / info / submit / finalize)
│   │   ├── submit_handler.py   # 7-phase submit lifecycle
│   │   ├── env_runner.py       # Run one episode against a Policy
│   │   ├── sandbox.py          # multiprocessing(spawn) + signal.setitimer act timeout
│   │   ├── heldout.py          # Held-out evaluation (called by Server.finalize)
│   │   ├── scoring.py          # final_score + auxiliary metrics (AUC, etc.)
│   │   ├── feedback.py         # summary.json / trajectory.jsonl / error.txt writers
│   │   └── seed_manager.py     # train.json / heldout.json → env_instance ID resolver
│   └── envs/
│       ├── __init__.py
│       ├── registry.py         # register_env(), get_env(), EnvDefinition
│       └── pendulum/
│           ├── __init__.py     # register_env(id="pendulum", ...) + factory
│           ├── TASK.md         # env-specific task description (served via GET /task)
│           ├── train.json      # {"real_seeds": [s0, …, s255]}
│           └── heldout.json    # {"real_seeds": [h0, …, h255]}
├── hlbench_cli/                # CONSUMER: argparse CLI (HTTP client only)
│   ├── __init__.py
│   └── main.py                 # hlbench {init,serve,info,submit,finalize}
├── agents/
│   └── pd_pendulum/
│       └── policy.py           # Reference agent (drop into workspace/system/)
├── tests/                      # 77 tests; gymnasium-skipped where not relevant
│   ├── test_skeleton.py        # Day 1
│   ├── test_env_runner.py      # Day 4
│   ├── test_sandbox.py         # Day 5
│   ├── test_submit_handler.py  # Day 6
│   ├── test_server_e2e.py      # Day 7 + 8
│   ├── test_scoring.py         # Day 11 (unit)
│   ├── test_http_server.py     # Day 9
│   └── test_cli.py             # Day 9
├── scripts/
│   ├── gen_seeds.py            # Generate train.json / heldout.json from master_seed
│   └── calibration.py          # Day 12-13 budget sweep
├── docs/
│   ├── architecture.md         # This file
│   ├── output.md               # runs/<...>/ layout
│   ├── submit-protocol.md      # 7 phases, 11 verdicts
│   ├── quickstart.md           # User walkthrough
│   └── findings.md             # Day 14 calibration analysis
├── README.md / AGENTS.md / SPEC.md / CLAUDE.md
└── pyproject.toml
```

**Deviations from the original sketch**:

- No `src/hlbench/cli/` or `src/hlbench/reference_agent/` packages —
  per CLAUDE.md invariant 9 (lib/consumer separation), the CLI and
  reference agents live outside `src/hlbench/`.
- `src/hlbench/http_server.py` is stdlib (`http.server`) rather than
  FastAPI/uvicorn — the original sketch named FastAPI but those
  packages flaked at install time and the protocol is small enough
  that the stdlib server is easier to maintain.
- No `policy_loader.py` — the original sketch had a separate module
  for importing `system/policy.py` and instantiating `Policy`, but
  that work happens inside the sandbox subprocess so it's all in
  `sandbox.py:_child_main`.
- Added `core/heldout.py` and `core/scoring.py` (Days 8 & 11) for
  finalize-time logic, which weren't broken out in the original
  sketch.
```

## 4. Component Sketches

### 4.1 `hlbench.core.server.Server`

Top-level entry; one instance per run.

```python
class Server:
    def __init__(
        self,
        env_id: str,
        workspace_dir: Path,
        episode_budget: int = 256,
        config_overrides: dict | None = None,
    ):
        """Initialize a per-run server.

        - Loads env definition (from registry)
        - Loads train.json and heldout.json
        - Creates workspace/ directory structure
        - Writes AGENTS.md to workspace; serves env's TASK.md via GET /task
        - Initializes state (remaining_budget, submit_count, etc.)
        """
        ...

    def info(self) -> dict:
        """Return full /info content (static + state). Used at run start
        and to refresh state after each submit."""
        ...

    def submit(self, env_instances: list[int]) -> SubmitResult:
        """Synchronously run one submit.

        Delegates to SubmitHandler. Returns when complete (after writing
        feedback files to workspace).
        """
        ...

    def finalize(self) -> RunFinalResult:
        """Trigger held-out evaluation, compute final_score, write run.json.
        Server is 'done' after this; subsequent submits raise.
        """
        ...
```

### 4.2 `hlbench.core.submit_handler.SubmitHandler`

Implements the 7-phase lifecycle (see `docs/submit-protocol.md §2`).

For MVP, simplifies:
- Phase 1: validate env_instance range (skip budget bound checks beyond N <= remaining)
- Phase 2 (snapshot): copy `workspace/system/` to a temp dir
- Phase 3 (validate): check `policy.py` exists, has `Policy` class (skip import scan)
- Phase 4 (compile): in subprocess, `import policy`
- Phase 5 (init): construct `Policy(obs_space, action_space, env_meta)`
- Phase 6 (execute): run `len(env_instances)` episodes via `EnvRunner`
- Phase 7 (commit): write `summary.json` atomically

```python
class SubmitHandler:
    def handle(
        self,
        env_instances: list[int],
        snapshot_dir: Path,
        feedback_dir: Path,
        env_factory: Callable,
        seed_manager: SeedManager,
    ) -> SubmitOutcome:
        """Returns SubmitOutcome with status, episode results, etc."""
        ...
```

### 4.3 `hlbench.core.env_runner.EnvRunner`

Runs one episode given a policy and a real seed.

```python
def run_episode(
    policy: Policy,
    env,
    real_seed: int,
    max_steps: int,
    record_obs: bool = True,
) -> EpisodeRecord:
    """One reset-to-termination cycle. Returns trajectory + return + length."""
    obs, info = env.reset(seed=real_seed)
    policy.reset(episode_index=0)  # MVP: pass 0 always
    traj = []
    total_reward = 0.0
    for t in range(max_steps):
        action = policy.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        traj.append({
            "t": t,
            "obs": obs.tolist() if record_obs else None,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": dict(info),
        })
        total_reward += reward
        obs = next_obs
        if terminated or truncated:
            break
    policy.on_episode_end(total_reward)
    return EpisodeRecord(trajectory=traj, return_=total_reward, length=t+1)
```

### 4.4 `hlbench.core.sandbox.Sandbox`

MVP sandbox: subprocess with basic rlimit + `signal.alarm` for `act()`.

```python
class Sandbox:
    def __init__(self, snapshot_dir: Path, env_factory, env_meta: dict):
        """Start a subprocess that imports system/policy.py and runs episodes
        on request. Communicate via pipe (pickle) for MVP.
        """
        ...

    def init_policy(self) -> None:
        """Send 'init' command to subprocess. Returns when Policy() done
        or raises on init_error / init_timeout."""
        ...

    def run_episode(self, real_seed: int) -> EpisodeRecord:
        """Send 'run episode' command. Subprocess steps env, returns trajectory."""
        ...

    def close(self) -> None:
        """Terminate subprocess."""
        ...
```

**MVP simplifications**:
- No `denied_import` enforcement (assume agent uses only allowed libs)
- No network blocking (agent code doesn't network for Pendulum)
- `act()` timeout via subprocess `signal.alarm(0.01)` from inside the subprocess
- Memory limit via `resource.setrlimit(RLIMIT_AS, 1_073_741_824)` at subprocess start

These are **real spec items** to add later. MVP works without them
because Pendulum + PD controller is well-behaved.

### 4.5 `hlbench.envs.registry`

Module-level registry, populated at import time.

```python
_REGISTRY: dict[str, EnvDefinition] = {}

@dataclass
class EnvDefinition:
    env_id: str
    env_version: str
    factory: Callable[[], gymnasium.Env]
    obs_space: dict
    action_space: dict
    max_episode_steps: int
    expert_baseline: float
    random_baseline: float
    n_env_instances: int = 256
    obs_storage: str = "inline"
    reward_components: dict[str, str] | None = None
    train_seeds_path: Path
    heldout_seeds_path: Path

def register_env(
    id: str,
    version: str,
    factory: Callable,
    expert_baseline: float,
    random_baseline: float,
    train_seeds: list[int],
    heldout_seeds: list[int],
    **kwargs,
) -> None:
    """Register an env. Typically called from envs/<id>/__init__.py."""
    ...

def get_env(env_id: str) -> EnvDefinition:
    """Look up by ID. Raises if unknown."""
    ...
```

### 4.6 `hlbench.envs.pendulum.env`

```python
# src/hlbench/envs/pendulum/env.py
import json
from pathlib import Path
import gymnasium
from hlbench.envs.registry import register_env

HERE = Path(__file__).parent
_TRAIN = json.load(open(HERE / "train.json"))["real_seeds"]
_HELDOUT = json.load(open(HERE / "heldout.json"))["real_seeds"]

def _factory():
    return gymnasium.make("Pendulum-v1", render_mode=None)

register_env(
    id="pendulum",
    version="0.1",
    factory=_factory,
    obs_space={"type": "Box", "shape": [3], "low": [-1, -1, -8], "high": [1, 1, 8]},
    action_space={"type": "Box", "shape": [1], "low": [-2.0], "high": [2.0]},
    max_episode_steps=200,
    expert_baseline=-150.0,    # internal, not exposed
    random_baseline=-1500.0,   # internal, not exposed
    train_seeds=_TRAIN,
    heldout_seeds=_HELDOUT,
    n_env_instances=256,
    obs_storage="inline",
    reward_components=None,    # Pendulum has scalar reward, no components
)
```

### 4.7 `hlbench.cli`

Minimal CLI:

```bash
hlbench init --env pendulum --dir ./my_run
  # Creates ./my_run/{AGENTS.md, system/policy.py.template, feedback/}; TASK.md served via /task
  # Internally: instantiates Server, persists workspace

hlbench info
  # Calls server.info(), pretty-prints JSON

hlbench submit --env-instances 0-7
  # Calls server.submit(list(range(8))), waits, prints summary

hlbench finalize
  # Calls server.finalize(), prints final_score
```

For MVP, `init` instantiates a `Server`, persists its state to a
JSON file in workspace; subsequent commands load that state.
(Quick-and-dirty persistence — not the long-term solution.)

## 5. Build Sequence (2-week plan)

### Week 1: Core machinery

**Day 1: Skeleton**
- `pyproject.toml`, package layout
- Empty modules with docstrings
- Pendulum env file (no logic yet)
- Test harness (pytest)

**Day 2: Env registry + Pendulum**
- Implement `register_env()`, `get_env()`
- Pendulum factory works (`gymnasium.make("Pendulum-v1")` runs)
- Generate `train.json` / `heldout.json` via `scripts/gen_seeds.py`
- Test: `get_env("pendulum")` returns valid EnvDefinition

**Day 3: SeedManager**
- Load JSON files, map env_instance ID → real_seed
- Validate ID range
- Test: known mapping is correct

**Day 4: EnvRunner**
- `run_episode()` works in-process (no sandbox yet)
- Hand-written PD controller as test Policy
- Test: PD on Pendulum gets reasonable return (>-300)

**Day 5: Sandbox subprocess**
- Start subprocess, run Policy.__init__ + reset + N steps
- Pickle communication
- act() wall-time enforcement
- Test: timeout actually triggers; init_error captured

**Day 6: SubmitHandler integration**
- Combine: validate input → spawn sandbox → run episodes → collect → write feedback
- summary.json + trajectory.jsonl actually written
- Test: end-to-end smoke (submit 4 episodes of PD on Pendulum, feedback files appear)

**Day 7: First end-to-end run**
- Wire up Server class
- Write reference PD agent, submit, iterate
- Goal: 5 submits, each shows feedback, mean_return ~ -200 (good PD)

### Week 2: Round out

**Day 8: Held-out evaluation**
- `finalize()` runs held-out seeds through final Policy
- Compute mean held-out return + normalized score
- Write run.json

**Day 9: CLI**
- `hlbench init`, `submit`, `info`, `finalize` actually work
- Workspace persistence (server state → JSON file)

**Day 10: Edge cases**
- invalid_env_instance (request ID 999) → verdict, no budget consumed
- init_error (Policy raises in init) → captured, traceback in errors.txt
- act_error / act_timeout (Policy.act raises / hangs) → per-episode error.txt

**Day 11: Auxiliary metrics**
- AUC, episodes_to_50pct, episodes_to_80pct
- All written to run.json

**Day 12-13: Calibration**
- Run multiple budgets (16, 64, 128, 256) with the same PD policy
- See how returns / scores vary
- Document findings

**Day 14: Iterate spec based on findings**
- What did we discover about default values?
- What feedback fields turned out unnecessary?
- Update SPEC.md based on real implementation feedback

## 6. Tech Stack

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | Standard for benchmarks |
| RL env | Gymnasium 0.29+ | Standard, has Pendulum-v1 |
| Numerics | numpy | Standard |
| HTTP server | `fastapi` + `uvicorn` | Minimal, modern, easy to test |
| HTTP client (in agent harness) | `httpx` | Modern, both sync/async APIs |
| Subprocess IPC (server ↔ sandboxed policy) | `multiprocessing.Pipe` + pickle | Stdlib, simple |
| Time limits | `signal.alarm` (inside subprocess) | Posix, stdlib |
| Memory limits | `resource.setrlimit` | Posix, stdlib |
| Serialization | `json` for config / summary; `jsonl` for trajectory | Human-readable |
| Testing | `pytest` | Standard |
| CLI | `argparse` (or `click` if more ergonomic) | Stdlib |
| Type hints | Standard typing | Modern Python |

**Notable absences**: no `docker`, no `gymnasium-robotics`, no
torch/jax. Add when needed.

## 7. Key Decisions Recorded

### 7.1 Sync HTTP submit

Pendulum episode is ~5 seconds wallclock. 8 episodes = ~40 seconds.
Local HTTP, no timeout pressure even with longer envs later.
**Sync POST /submit blocks until done** — no async/long-poll needed.

### 7.2 Sandbox uses subprocess + pickle

Alternatives considered:
- Direct in-process: no isolation, agent code can crash server
- Threading: GIL issues, no rlimit per thread
- Docker per submit: too heavy for MVP
- **subprocess + pickle**: ✓ stdlib, simple, gives process isolation

Pickle is fine because both ends are trusted hlbench code; agent
policy is loaded by subprocess from `policy.py`.

### 7.3 Static seed files as JSON arrays

Format: `{"real_seeds": [int, int, ...]}`. Loaded once at env
registration. Bit-exact reproducible across machines.

`scripts/gen_seeds.py` generates from a master seed:
```python
import json, random
def gen(master_seed: int, n: int) -> list[int]:
    rng = random.Random(master_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]
```

### 7.4 Server state persistence in MVP

Server holds in-memory state during a run. To survive between CLI
invocations, persist to `workspace/.server_state.json` (private file,
not part of agent-facing workspace). This is a temporary hack —
proper solution is to have CLI run server long-lived.

### 7.5 No checkpoints/ for MVP

`runs/.../checkpoints/submit_NNN/` (per-submit code snapshot) is in
the full spec but not needed for MVP. Add when implementing replay.

### 7.6 No agent.jsonl for MVP

Logging agent harness events to `logs/agent.jsonl` is a future
feature. For MVP, just print to stderr.

## 8. Implementation Risks

### 8.1 Pickle subprocess communication

Big trajectories (1000 steps × small obs) could be slow via pickle.
**Mitigation**: profile early. If slow, switch to shared memory or
just write trajectory.jsonl from inside subprocess.

### 8.2 act() timeout via signal.alarm

`signal.alarm` requires main thread. Subprocess is single-threaded
so OK. But interaction with numpy/scipy that mask signals is a known
issue. **Mitigation**: have a thread monitor wall time as fallback;
SIGKILL after 2x deadline.

### 8.3 Workspace persistence between CLI calls

If user calls `hlbench submit` twice, the server instance dies
between calls. We need to either:
- Restart server from persisted state (current plan: JSON file in workspace)
- Run a daemon

For MVP we use the JSON file approach. **Risk**: race conditions if
user invokes commands concurrently. For MVP, assume single-user
serial usage.

### 8.4 Held-out determinism

Pendulum's `env.reset(seed=X)` should give deterministic initial state.
Verify this empirically before relying on it.

## 9. Validation Checklist

By end of Week 2, all of the following should pass:

- [ ] `pip install -e .` succeeds, package imports cleanly
- [ ] `hlbench init --env pendulum --dir ./run1` creates workspace
- [ ] Workspace has correct files (AGENTS.md, system/, feedback/) and `GET /task` returns env's TASK.md content
- [ ] `hlbench info` returns valid JSON matching SPEC §1.1 schema
- [ ] PD controller policy written by hand, ~30 LOC
- [ ] `hlbench submit --env-instances 0-7` returns in <60s
- [ ] `workspace/feedback/submit_000/summary.json` matches SPEC §4.1
- [ ] `workspace/feedback/submit_000/episodes/ep_000/trajectory.jsonl` has 200 lines
- [ ] mean_return on PD policy is in range [-300, -100]
- [ ] `hlbench submit --env-instances 999` rejected with `invalid_env_instance`
- [ ] Bad policy (raises in init) gets `init_error` status
- [ ] `hlbench finalize` computes held-out final_score
- [ ] `runs/<...>/run.json` written with final_score in [0, 120]
- [ ] 3 budget calibration runs (16, 64, 256) complete, results recorded

## 10. After MVP: What's Next

Order of next features after Pendulum MVP works:

1. **Second env**: HalfCheetah (tests larger obs, longer episodes,
   reward_components)
2. **Third env**: Atari RAM (tests discrete actions)
3. **Sandbox hardening**: enforce denied_imports, network blocking
4. **CarRacing**: tests pixel obs + observations.npy sidecar
5. **Container deployment**: Docker for prod isolation

Items explicitly **not** on the roadmap (per shared-host commitment):
- Remote / distributed deployment
- Async submit + long-poll
- Tar bundle feedback transport

Each step exposes spec gaps to fix. Don't try to do everything at once.

## 11. Open Questions to Resolve During Implementation

These come up the moment you start coding:

1. **`Policy.reset(episode_index)` — what's `episode_index`?** Local in
   submit (0..n-1)? Global? Could be useful for policy to know it's
   the first episode of a submit batch. Decision: local, 0-based per
   submit.

2. **Action serialization for Box(1)**: Pendulum action is a 1-D float.
   In JSON: `[0.5]` (list of 1) or `0.5` (scalar)? Decision: always
   list per Gymnasium convention (`Box` is list of N floats).

3. **Workspace path conventions**: relative or absolute?
   Decision: server stores absolute internally, CLI accepts relative
   and resolves to absolute.

4. **AGENTS.md delivery**: ship as static file with package, copy on
   init. (No /agent endpoint for MVP.)

5. **TASK.md serving**: hand-write a Pendulum TASK.md inside the env package; server reads it on demand for `GET /task` (no longer staged into workspace).

## 12. References

- Protocol contract: `SPEC.md`
- Agent rules: `AGENTS.md`
- Run outputs: `docs/output.md`
- Submit lifecycle: `docs/submit-protocol.md`
- This implementation plan: `docs/architecture.md` (you are here)
