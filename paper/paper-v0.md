# HLBench-Pro: A Closed-Loop Benchmark for LLM-Driven Policy Synthesis with Held-Out Generalization

**Version 0.1 (preprint draft) — May 2026**

*Authors: TBD*

---

## Abstract

Most benchmarks for evaluating large language model (LLM) agents
measure single-shot capabilities — fixing a bug given a test suite
(SWE-bench), writing a script given a Kaggle competition specification
(MLE-bench), or answering a question given a multi-tool harness
(GAIA, AgentBench). None measure the specific capability we believe
matters most for real-world deployment: **iteratively refining
policy code based on rich step-level environment feedback, under a
tight rollout budget, with strict held-out generalization**.

We introduce **HLBench-Pro**, a closed-loop benchmark that fills
this gap. The benchmark scores agents on how well their final policy
code generalizes to a hidden seed pool, after a fixed budget of
in-loop episodes during which the agent edits code and submits new
versions. The protocol enforces three design pressures: (1) policy
synthesis must be the bottleneck (not knowledge recall), (2) held-out
generalization must dominate scoring (not in-loop overfitting), and
(3) anti-cheat must be cryptographically rigorous (no held-out
visibility, no pretrained-policy shortcuts).

This version (v0.1) describes the protocol, the v1 environment
suite (16 envs across 6 categories — 6 currently implemented and
landed in the open-source release), and the design decisions behind
each. We position HLBench-Pro relative to SWE-bench (one-shot fix),
MLE-bench (one-shot script), Voyager (closed-loop but
single-environment), and Eureka (closed-loop reward design). The
benchmark is open-source and ready for frontier-model evaluation;
full empirical results across multiple frontier LLMs are deferred
to v1.0.

---

## 1. Introduction

Real-world deployment of LLM-driven autonomous systems involves
**iterative refinement of agent code** based on **environmental
feedback** under **resource constraints**. A typical engineering
workflow looks like: (1) deploy initial policy, (2) observe
trajectories from real or simulated runs, (3) diagnose failure modes,
(4) edit code, (5) re-deploy, (6) repeat until performance plateaus
or budget is exhausted. The capability the system author needs is
not "write the right code on the first try" but "extract maximum
useful information per evaluation rollout and translate that into
incremental code improvements."

No widely-used LLM benchmark measures this composite capability.
SWE-bench [Jimenez et al., 2023] tests one-shot patch synthesis
against a hidden test suite — there is no iteration loop. MLE-bench
[Chan et al., 2024] tests one-shot ML pipeline construction against
a Kaggle leaderboard — again no iteration with environmental
feedback. AgentBench [Liu et al., 2023] and GAIA [Mialon et al.,
2023] test multi-step tool use, but the loop is "tool returns →
agent reads → agent calls next tool" rather than "agent's deployed
artifact runs → produces feedback → agent edits artifact." Voyager
[Wang et al., 2023] does test closed-loop iteration in a Minecraft
environment, but is single-task and lacks held-out generalization.
Eureka [Ma et al., 2024] iterates LLM-written reward functions
against an RL training loop, but the LLM never writes the policy
itself.

The gap is real and structural: **closed-loop, budget-constrained,
LLM-driven policy synthesis with held-out generalization**. Filling
it requires a benchmark protocol that addresses three coupled
problems:

1. **Iteration must be load-bearing.** If a textbook controller
   solves the task on the first attempt, the benchmark measures
   pretraining knowledge, not iteration. Every environment in the
   suite must satisfy "policy synthesis is the bottleneck."

2. **Held-out must be invisible.** If the agent sees held-out seeds
   or aggregate held-out scores during the run, it can hill-climb
   on those metrics rather than the underlying capability. The
   benchmark protocol must guarantee zero held-out leakage at every
   API surface.

3. **The score function must be normalized and method-neutral.**
   Hand-coded heuristics, classical control, search-and-planning,
   small networks trained from scratch, and any combination must
   all be admissible. The benchmark scores the resulting policy's
   performance on held-out, not the methodology used to produce it.

We introduce **HLBench-Pro**, a closed-loop benchmark protocol that
addresses all three. We release this work as a v0.1 preprint
covering the protocol design, the v1 environment suite, the
reference implementation, and the design rationale behind every
non-trivial choice. Full frontier-model empirical results are
deferred to v1.0; the v0.1 release establishes the protocol
contract and the env roster.

**Contributions**:

- A complete benchmark protocol (workspace layout, four HTTP
  endpoints, Policy interface, feedback schema, scoring) designed
  for closed-loop iteration with rigorous anti-cheat.
- A v1 environment roster of 16 envs across 6 categories
  (visual control, procedural visual generalization, spatial
  reasoning, visual game, online algorithms, hardcore state-based
  control), 6 of which are landed in the v0.1 release.
- An open-source reference implementation: server library, HTTP
  wrapper, sandbox, scoring, automated Claude Code driver, and a
  150+ test suite enforcing the protocol contract.
- Design analysis: which design choices were forced by the three
  problems above, and which alternatives we rejected and why
  (Section 7.2).

---

## 2. Related Work

We compare HLBench-Pro along the four axes that distinguish it from
existing benchmarks: closed-loop iteration, environmental feedback
richness, code iteration depth, and held-out generalization.

| Benchmark | Closed loop? | Env feedback | Code iteration | Held-out |
|---|---|---|---|---|
| SWE-bench | no | test pass/fail | one-shot fix | hidden test suite |
| SWE-bench Verified | no | test pass/fail | one-shot fix | curated 500 |
| MLE-bench | no | validation score | one-shot script | Kaggle private LB |
| AgentBench / GAIA | partial | tool returns | varies | per-task |
| HumanEval / MBPP | no | none | none | n/a |
| BIG-bench Hard | no | none | none | task-level |
| Voyager | yes | rich (Minecraft) | yes | none (single env) |
| Eureka | yes | RL training curves | reward only | env-level |
| RL benchmarks (Atari, MuJoCo) | n/a | dense rewards | n/a | n/a (algos not agents) |
| **HLBench-Pro** | **yes** | **per-step trajectories + stdout** | **yes, budgeted** | **disjoint hidden seed pool** |

**SWE-bench [Jimenez et al., 2023]**: 2294 real GitHub bug-fix tasks
extracted from 12 popular Python repositories. Tests one-shot patch
synthesis against the existing test suite. No iteration; the agent
either fixes the bug or it doesn't. SWE-bench Verified [OpenAI, 2024]
is the 500-instance hand-validated subset. SWE-bench's strength is
task source — it scales with GitHub.

**MLE-bench [Chan et al., 2024]**: 75 Kaggle competitions, with
agents producing complete ML pipelines (data loading, model training,
prediction generation) in a fixed compute budget. Scoring uses
Kaggle's medal thresholds. Like SWE-bench, MLE-bench's strength is
task source: Kaggle provides a continuous stream of well-curated
real-world ML problems.

**AgentBench [Liu et al., 2023] and GAIA [Mialon et al., 2023]**:
multi-tool agentic environments. The agent has tool calls (web
search, code execution, file IO) and produces an answer. The loop
is "tool returns → agent thinks → agent calls next tool" — useful
but distinct from "agent writes deployable artifact and observes
the artifact's runtime behavior."

**Voyager [Wang et al., 2023]**: LLM-driven skill acquisition in
Minecraft. The closest existing work to our setting in spirit.
Voyager iteratively writes Minecraft skill code, observes execution
results, and refines. Two key differences from HLBench-Pro: (a)
Voyager is single-environment (Minecraft), so generalization
testing is limited to within-Minecraft skill diversity; (b) there
is no formal held-out evaluation — performance is reported on
in-loop trajectories.

**Eureka [Ma et al., 2024]**: LLM-driven reward function synthesis
for RL training. The LLM writes reward code, an RL agent trains
under that reward, the LLM observes training curves, refines.
Eureka is closed-loop and demonstrates that LLM-written code can
iterate productively; however, the LLM does not write the
*policy* — that is the RL agent's job. HLBench-Pro inverts this:
the LLM writes the policy directly.

**Reflexion [Shinn et al., 2023] and Self-Refine [Madaan et al.,
2023]**: foundational papers on iterative LLM refinement. These
provide methods, not benchmarks, but the iteration paradigm they
established is what HLBench-Pro is designed to evaluate.

**BIG-bench Hard [Suzgun et al., 2022]**: 23 hard tasks curated
from BIG-bench, demonstrating that smaller, curated benchmarks
can produce strong findings if calibrated for frontier-model
discrimination. We follow this precedent at the env-suite level
(16 envs, deliberately curated rather than batch-collected).

---

## 3. Benchmark Design

### 3.1 Protocol overview

HLBench-Pro is a **per-run server** architecture: one Python
process per `(model, env, exp_id)` triple, exposing four HTTP
endpoints. The agent and server share a local filesystem
(workspace directory) and communicate via:

- `GET /info` — effective configuration + dynamic state
  (remaining budget, last submit status). Single source of truth
  for run config.
- `GET /task` — env-specific task description as
  `text/markdown`, served on demand.
- `POST /submit` — synchronously runs N episodes on requested
  env_instance IDs. Returns submit-level summary; per-episode
  artifacts (trajectories, errors, captured stdout/stderr)
  written directly to the shared workspace.
- `POST /finalize` — declares run finished. Triggers held-out
  evaluation. The agent's most recent successful submit is the
  final policy.

The HTTP layer is a thin stdlib wrapper (no FastAPI dependency)
over a Python library (`hlbench.core.Server`); tests and tooling
use the lib directly, but agents *must* use HTTP — this enforces
the contract that the agent has no privileged access to server
internals.

### 3.2 Workspace contract

The agent's workspace contains exactly three top-level entries:

- `AGENTS.md` — protocol rules document (read-only, shipped at
  run start).
- `system/` — a Python package the agent writes freely.
  `system/policy.py` is required and must define a `Policy` class
  conforming to the interface in Section 3.3. The agent may
  organize submodules anywhere under `system/`.
- `feedback/submit_NNN/` — populated by the server after each
  submit, containing `summary.json` and per-episode subdirectories
  with trajectory JSONL, captured stdout/stderr, and any error
  files.

Notably absent: any run-config file, any task-description file,
or any held-out information. Run configuration is served by
`/info`; task description by `/task`. Both are server-only,
preventing the agent from caching stale views or modifying its
view of the rules.

### 3.3 Policy interface

```python
class Policy:
    def __init__(
        self,
        obs_space: Mapping[str, Any],
        action_space: Mapping[str, Any],
        env_meta: Mapping[str, Any],
    ) -> None: ...

    def reset(self, episode_index: int) -> None: ...

    def act(self, obs) -> Any: ...
```

Three methods: `__init__` is called once per submit and may load
files from `system/`; `reset(i)` is called at the start of each
of the submit's episodes; `act(obs)` is called once per step and
must return a valid action.

The interface deliberately omits an `on_episode_end` hook.
Episode-level signals reach the agent through
`feedback/.../summary.json:returns` after the submit completes,
which is sufficient for an LLM-driven outer loop. Adding a
within-submit hook would only duplicate that channel and muddy
attribution between "what the agent decided" and "what the
policy learned."

### 3.4 Env-instance ID indirection

Agents address env instances by **integer ID** in
`[0, n_env_instances)`. Internally, each ID maps to a real seed
loaded from a per-env static `train.json` file. The mapping is
server-internal and **never exposed**. Two consequences:

1. **Reproducibility**: env_instance ID 5 always corresponds to
   the same real seed under a given env_version. Cross-run
   comparisons are bit-exact.
2. **Anti-cheat**: agents cannot enumerate or guess held-out
   seeds, since they only ever see integer IDs in the train pool
   range.

The held-out seed pool lives in a separate `heldout.json` file,
drawn from a disjoint range. Held-out is run exactly once at
finalize, on the final policy snapshot.

### 3.5 Scoring

The headline score is

```
final_score = clip(
    (mean_held_out − random_baseline) /
    (expert_baseline − random_baseline),
    0, 1.2
) × 100
```

Both `random_baseline` (mean return of a uniform random policy
over the same M held-out episodes) and `expert_baseline`
(published expert-level performance per env) are env-internal
and **never exposed to the agent during or after the run**.
This is critical: an agent that sees the expert threshold can
hill-climb on the threshold; an agent that doesn't must
optimize raw return without targeting a known number, which is
closer to the realistic deployment setting.

The `clip(0, 1.2)` lets super-expert performance register but
bounds outliers. Multiplying by 100 gives the familiar "0–120"
percentage scale.

Auxiliary metrics — `auc_in_loop`, `episodes_to_50pct`,
`episodes_to_80pct`, `held_out_gap`, `n_submits`, `episodes_used`
— are reported alongside but do not affect the headline.

### 3.6 Sandbox + anti-cheat

The agent's policy code runs in a `multiprocessing(spawn)` child
process per submit, with:

- A `sys.meta_path` import hook enforcing
  `denied_imports = {transformers, huggingface_hub, timm,
  diffusers, openai, anthropic, google.genai, cohere,
  stable_baselines3, ray, rllib, cleanrl, sb3_contrib, urllib,
  requests, socket, httpx, aiohttp, subprocess, os.system, ...}`.
  Importing any blocks the submit with the `denied_import`
  verdict. The denied list is exposed via `/info` so agents
  cannot legitimately stumble into a denied module.
- `signal.setitimer(ITIMER_REAL)` enforcing the per-`act()`
  wall-time limit (default 10 ms; env-overridable upward for
  search-heavy heuristic envs).
- A submit-level wall-time cap (default 300 s) and best-effort
  RSS cap (`resource.setrlimit(RLIMIT_AS)`).
- Per-episode `stdout.txt` / `stderr.txt` capture, allowing the
  policy to `print()` diagnostic information that reaches the
  agent's next iteration.

Eleven submit-level verdicts (in v0.1; ten after the
`oversize` deletion in this release) form a unified enum
covering every failure mode, with strict mutual-exclusion
guarantees between submit-level errors and per-episode errors.
The verdict system is documented in `docs/submit-protocol.md`.

---

## 4. Environment Suite

### 4.1 Composition

The v1 roster comprises 16 environments across 6 categories,
calibrated for frontier-model discrimination:

| # | Env | Category | Obs type |
|---|---|---|---|
| 1 | CarRacing-Pixel | Visual control | pixel 96×96×3 |
| 2 | MiniGrid-Pixel-Hard | Visual control | pixel partial-view |
| 3 | Pendulum-from-Pixels | Visual control | pixel 64×64×3 |
| 4 | Procgen-CoinRun | Procedural visual | pixel 64×64×3 |
| 5 | Procgen-BigFish | Procedural visual | pixel 64×64×3 |
| 6 | Procgen-Maze | Procedural visual | pixel 64×64×3 |
| 7 | Sokoban-Pixel | Spatial reasoning | pixel grid |
| 8 | Hidden-Maze | Spatial reasoning | pixel revealed-map |
| 9 | Snake-Pixel | Visual game | pixel grid |
| 10 | Frogger-Mini | Visual game | pixel grid |
| 11 | Cache-Replacement | Online algorithm | state |
| 12 | K-Server | Online algorithm | state |
| 13 | Online-Bipartite-Matching | Online algorithm | state |
| 14 | Pendulum-Hardcore | Hardcore control | state Box(3) |
| 15 | Lunar-Hardcore | Hardcore control | state Box(8) |
| 16 | Bipedal-Hardcore | Hardcore control | state Box(24) |

10 of the 16 envs use pixel observations; 6 use state-based
observations. The state-based envs are deliberate: they form
the cross-paradigm comparison group that supports the paper
finding "frontier models score X on visual envs, Y on
state-based, the gap measures visual reasoning capability."

### 4.2 Currently implemented (v0.1)

Six envs are landed in the open-source v0.1 release:

**Hardcore state-based control (Gymnasium-based, #14-16):**

- `pendulum_hardcore` (#14) — `gym.Wrapper` over Pendulum-v1
  reassigning `(mass, length, gravity)` per `reset(seed=)` from
  train (`m∈[0.5,2.0], l∈[0.7,1.5], g∈[8.0,12.0]`) or held-out
  (`m∈[2.0,3.5], l∈[1.5,2.2], g∈[4.0,8.0]`) ranges based on
  seed magnitude. Disjoint OOD held-out tests adaptive control
  vs. fixed gains.

- `lunar_hardcore` (#15) — `gym.Wrapper` over
  LunarLanderContinuous-v3 with `enable_wind=True`, reassigning
  `wind_power` and `turbulence_power` per seed from train vs.
  disjoint held-out OOD ranges.

- `bipedal_hardcore` (#16) — direct `gymnasium.make
  ("BipedalWalkerHardcore-v3")`. Built-in procedural terrain
  (stumps, ladders, pits) provides per-seed variation.

**Online algorithms (custom pure-Python, #11-13):**

- `cache_replacement` (#11) — capacity-8 cache, 64 object IDs,
  trace length 500. Train: Zipfian (LRU-friendly). Held-out:
  scan-heavy (LRU-hostile).

- `k_server` (#12) — 3 servers on `[-1,1]^2`, 200 requests per
  episode. Train: 2-Gaussian mix. Held-out: 4 corners with 75%
  weight on one corner, defeating greedy.

- `online_bipartite_matching` (#13) — 16 left vertices, 24
  online arrivals. Train: random `G(N,M,p=0.25)`. Held-out:
  KVV-style adversarial structure with "honey trap" edges.

### 4.3 Pending (10 visual envs)

The 10 visual envs (#1–10) require a per-step external
observation-storage mechanism (`observations.npy` side-car
file) that is not yet implemented. The protocol's
`obs_storage: "external"` mode is fully specified
(SPEC §1.1, §4.6) but the writer/reader is deferred. Visual
envs are the primary deliverable for v1.0.

### 4.4 Design principles

Every env in the roster must satisfy two tests:

1. **Policy synthesis is the bottleneck.** A textbook
   algorithm exists for many of these problems (LRU for cache,
   nearest-neighbor for k-server, RANKING for online matching,
   PD/LQR for control). But the bench's held-out distributions
   are designed such that the textbook algorithm scores well on
   train and poorly on held-out. The agent must extend or
   combine textbook approaches to generalize.

2. **Held-out generalization must matter.** Train and held-out
   pools are drawn from disjoint distributions on every env. An
   agent that hill-climbs on in-loop returns will demonstrably
   collapse on held-out, distinguishing real generalization from
   in-loop overfitting.

A capability matrix maps each env to the agent capabilities it
probes (vision, memory, search, online decision, generalization,
code complexity); every capability has at least 4 envs probing
it (envs.md §"Cross-cutting capability matrix"), satisfying the
per-category statistical-power floor for capability-level
findings in the empirical study.

---

## 5. Experimental Setup

### 5.1 Reference implementation

The v0.1 release is a complete, open-source reference
implementation:

- **`hlbench/core/`** — server library: per-run `Server` class,
  `SubmitHandler` (7-phase submit lifecycle, 10 verdicts),
  `Sandbox` (spawn-process child holding one Policy instance),
  `env_runner` (single-episode runner with SPEC §4.2 trajectory
  schema), `heldout` (final evaluation), `scoring` (normalized
  score + auxiliary metrics), `feedback` (atomic JSON writers).
- **`hlbench/http_server.py`** — stdlib `http.server` HTTP
  wrapper exposing `GET /info`, `GET /task`, `POST /submit`,
  `POST /finalize`.
- **`hlbench_cli/`** — CLI: `hlbench {init, serve, info,
  submit, finalize, agent}`.
- **`hlbench_harness/`** — automated Claude Code driver:
  drives `claude --print --resume <session_id>` in a closed
  loop, preserves conversation context across iterations,
  emits `agent.jsonl` activity log per `output.md §6.2`.
- **6 environments** registered (3 hardcore + 3 online algorithm)
  plus 5 v0 baseline envs (pendulum, acrobot,
  mountain_car_continuous, bipedal_walker, lunar_lander_continuous)
  retained for backward compatibility and integration testing.
- **201 unit and integration tests** under
  `pytest -q + ruff + mypy --strict` discipline.

### 5.2 Frontier-model evaluation (v1.0 deliverable)

The v0.1 release does not include a frontier-model evaluation
matrix. The infrastructure to run it is in place via
`hlbench agent`, but a complete matrix (5+ models × 16 envs)
costs roughly $1–3K USD in API credits and is gated on the
visual-env infrastructure described in Section 4.3.

The v1.0 paper will report:

- 5 frontier models (Claude 4.7, GPT-5, Gemini 2.5, etc.)
- All 16 envs (after observations.npy infrastructure lands)
- Per-env score distribution (`final_score`,
  `held_out_gap`, `auc_in_loop`)
- Capability-level findings via the cross-cutting matrix in
  Section 4.4
- Comparison of closed-loop iteration vs. one-shot baseline
  (the headline finding: does iteration help, by how much,
  and how does the help vary by capability?)

### 5.3 Reference baselines (v0.1)

A reference PD controller for `pendulum` (Pendulum-v1, the v0
gentle variant) hits `final_score = 98.3`, with held-out mean
return = −168 against `random_baseline = −1200` and
`expert_baseline = −150` (`docs/findings.md`). This validates:

- The protocol end-to-end (init → submit → finalize → run.json).
- The held-out evaluation (mean return on hidden seeds is close
  to expert, suggesting the textbook PD controller does
  generalize well on the easy gentle Pendulum).
- The scoring pipeline (normalized score lands in the expected
  98–100 range for a near-expert policy).

We expect frontier-model scores on `pendulum_hardcore` (the v1
domain-randomized variant) to be substantially lower than on
`pendulum` — verifying that domain randomization successfully
breaks the textbook-PD ceiling.

---

## 6. Reference Baselines and Sanity Checks

This section reports the empirical sanity checks performed on
the v0.1 release. Full frontier-model results are deferred to
v1.0.

### 6.1 Pendulum-v1 (gentle): PD controller

- Method: energy-shaping swing-up + PD stabilization, ~30 lines
  of numpy.
- `final_score = 98.3` (across 256 held-out episodes).
- `held_out_gap = +8` (in-loop score 1252 vs. held-out 1244 in
  raw return, after normalization the gap is small) — validates
  the determinism of the protocol.

### 6.2 Trace distribution divergence (online algorithm envs)

For each of `cache_replacement`, `k_server`,
`online_bipartite_matching`, we verify that the train and
held-out distributions are quantitatively divergent:

- **`cache_replacement`**: in the first 100 accesses of a
  representative trace, train (Zipfian) has 21 unique IDs
  while held-out (scan) has 64 (max possible). A pure-LRU
  policy is trivially defeated on held-out by this structural
  difference.
- **`k_server`**: held-out concentrates 75% of requests in the
  upper-right quadrant `[0.5,1]×[0.5,1]`; train concentrates
  ~0%. A greedy-nearest policy that "earns" a server in the
  hot quadrant during the first few requests overcommits and
  pays high travel costs on the cold-quadrant 25%.
- **`online_bipartite_matching`**: held-out has zero edges from
  the first 12 right-arrivals to the right half of left
  vertices, vs. dozens in train — verifying the structural
  asymmetry.

These divergences are checked via per-env unit tests
(`tests/test_online_algo_envs.py`), ensuring the held-out
generalization claim is not vacuous.

### 6.3 Hardcore parameter randomization sanity

For `pendulum_hardcore` and `lunar_hardcore`, we verify that
1000 train seeds map to parameter values strictly within the
documented train ranges, and 1000 held-out seeds map to values
strictly within the disjoint held-out ranges
(`tests/test_hardcore_envs.py`). Disjointness on every parameter
axis is unit-tested.

`bipedal_hardcore` directly uses Gymnasium's
`BipedalWalkerHardcore-v3`, whose terrain procedural generation
is the standard Gymnasium implementation; no per-seed wrapper
is required.

### 6.4 Test suite

- 201 tests, all passing under `pytest -q`.
- mypy strict (`--strict`) clean across 39 source files.
- ruff clean.
- CI workflow (`.github/workflows/ci.yml`) runs all three on
  every push and PR.

---

## 7. Discussion

### 7.1 Why this benchmark exists at the size it does

A natural question for benchmark design is: why 16 envs, not
75? The answer is twofold.

First, env quality is more important than env quantity for
discrimination. MLE-bench has 75 envs, but the headline finding
("agents reach bronze on ~17% of tasks") is dominated by a few
tasks per category. Empirically, BIG-bench Hard demonstrated
that 23 carefully curated tasks can produce stronger findings
than 200 batch-collected tasks. We follow this precedent at the
env-suite level.

Second, every env in our roster requires substantial design
work: a held-out distribution disjoint from train, a
discrimination check (frontier models must score noticeably
differently), an anti-cheat check (no off-the-shelf solution in
the agent's allowed imports), and a calibration of expert/random
baselines. This cost is real, and 16 is what we believe is
achievable to do well in the time available.

If frontier models saturate the v1 suite (i.e., scores cluster
near 100 on most envs), we will swap distributions or expand
adversarial structure rather than expand env count. **Quality
of discrimination is the metric the suite is optimized for, not
breadth of coverage.**

### 7.2 Design decisions and rejected alternatives

Several design choices deserve explicit justification:

**Why HTTP, not Python lib, as the agent's only channel?** A
Python library API would let the agent introspect server state,
including held-out information, with sufficient reflection.
HTTP forces the contract: the server controls every byte the
agent sees. Tests and tooling use the lib; agents must use HTTP.

**Why a per-run server, not a multi-tenant server?** Per-run
servers eliminate cross-run contamination by construction.
Multi-tenancy would require careful state isolation; per-run is
simpler to verify and reason about.

**Why workspace = exactly 3 things?** Earlier drafts had 4
(adding `TASK.md` to the workspace). We removed `TASK.md` and
moved it to `GET /task` because: (1) it parallels how
`GET /info` replaced `_run.json`, (2) workspace minimalism
eliminates the "agent edited the rules" failure mode, (3) the
env package still ships the task description as a static file —
just not staged.

**Why no `expert_baseline` exposure?** An agent that knows the
expert threshold can hill-climb on the threshold rather than
the underlying capability. Forcing the agent to optimize raw
return without a target threshold is closer to realistic
deployment, where there is no oracle.

**Why drop the `oversize` verdict and `system/` size limits?**
The advisory size limits (50KB total / 25KB single file in
SPEC §4.3 v0.1.0a1) made search-heavy heuristic envs (Sokoban
deadlock detection + pattern databases, MCTS implementations)
infeasible. The trade-off was: enforce strict limits and lose
heuristic-flavored envs, or remove the cap and let envs choose
what's reasonable. We chose the latter (changelog
`[Unreleased]`).

### 7.3 Limitations

- **Six of 16 envs landed.** The 10 visual envs are gated on
  `observations.npy` infrastructure (~3 dev-weeks, separate
  work item). v0.1 paper claims the protocol; visual envs are
  the primary v1.0 deliverable.
- **No frontier-model results yet.** Empirical evaluation
  matrix (5+ models × 16 envs) is the v1.0 deliverable. The
  driver infrastructure (`hlbench agent`) is fully
  implemented and tested.
- **Single-author, single-organization.** Established benchmarks
  derive credibility partly from multi-organization adoption.
  v0.1 establishes the protocol; adoption is downstream.
- **No textbook-resistance proof.** While we argue
  qualitatively that "policy synthesis is the bottleneck" on
  every env, we have not empirically verified that frontier
  models cannot trivially solve these envs from pretraining
  knowledge. v1.0 will report this directly.

### 7.4 Future work

Beyond the v1.0 deliverables (visual envs + frontier-model
matrix), we plan:

- **Textbook-resistance ablation**: for each env, measure how
  much LLM performance degrades when we hide the env name and
  task description from the prompt. Larger degradation =
  stronger evidence that policy synthesis (not knowledge
  recall) is the bottleneck.
- **Iteration-helps ablation**: compare closed-loop performance
  (the standard run) vs. one-shot performance (a single submit
  with the full episode budget). Headline finding: does
  iteration help, by how much, on which categories?
- **Capability decomposition**: regress per-env score against
  the cross-cutting capability matrix (vision, memory, search,
  etc.) to identify which agent capabilities most strongly
  predict overall benchmark performance.
- **Public leaderboard**: once 5+ models have been evaluated,
  open a leaderboard at the project URL.

---

## 8. Conclusion

We have introduced HLBench-Pro, a closed-loop benchmark for
evaluating LLM-driven policy synthesis with held-out
generalization. The benchmark fills a real gap in the
evaluation landscape: SWE-bench / MLE-bench are one-shot,
agentic harnesses are tool-use rather than artifact-deployment,
and Voyager / Eureka are closed-loop but limited in
generalization scope or method coverage. HLBench-Pro is the
first benchmark to combine all four of: closed-loop iteration,
budget-constrained, rich step-level feedback, and disjoint
held-out generalization.

The v0.1 release covers the protocol (workspace, four HTTP
endpoints, Policy interface, scoring, anti-cheat sandbox), the
v1 environment roster (16 envs across 6 categories, 6 of which
are landed), and the design rationale behind every non-trivial
choice. v1.0 will close out the visual-env infrastructure and
report the full frontier-model evaluation matrix.

We release the open-source reference implementation and all
six v1 envs. The protocol is ready for community adoption now;
the empirical findings will follow in v1.0.

---

## References

- Jimenez, C., Yang, J., Wettig, A., Yao, S., Pei, K., Press,
  O., Narasimhan, K. "SWE-bench: Can Language Models Resolve
  Real-World GitHub Issues?" *ICLR 2024.*
- OpenAI. "SWE-bench Verified." 2024.
  https://openai.com/index/introducing-swe-bench-verified/
- Chan, J. S., et al. "MLE-bench: Evaluating Machine Learning
  Agents on Machine Learning Engineering." *2024.*
- Liu, X., et al. "AgentBench: Evaluating LLMs as Agents."
  *ICLR 2024.*
- Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y.,
  Scialom, T. "GAIA: a benchmark for General AI Assistants."
  *ICLR 2024.*
- Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu,
  Y., Fan, L., Anandkumar, A. "Voyager: An Open-Ended Embodied
  Agent with Large Language Models." *NeurIPS 2023 (Foundation
  Models for Decision Making Workshop).*
- Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O.,
  Jayaraman, D., Zhu, Y., Fan, L., Anandkumar, A. "Eureka:
  Human-Level Reward Design via Coding Large Language
  Models." *ICLR 2024.*
- Shinn, N., Cassano, F., Berman, E., Gopinath, A.,
  Narasimhan, K., Yao, S. "Reflexion: Language Agents with
  Verbal Reinforcement Learning." *NeurIPS 2023.*
- Madaan, A., et al. "Self-Refine: Iterative Refinement with
  Self-Feedback." *NeurIPS 2023.*
- Suzgun, M., et al. "Challenging BIG-Bench Tasks and Whether
  Chain-of-Thought Can Solve Them." *2022.*
- Cobbe, K., Hesse, C., Hilton, J., Schulman, J. "Leveraging
  Procedural Generation to Benchmark Reinforcement Learning."
  *ICML 2020.* (Procgen)
- Karp, R. M., Vazirani, U. V., Vazirani, V. V. "An Optimal
  Algorithm for On-line Bipartite Matching." *STOC 1990.*
- Koutsoupias, E., Papadimitriou, C. H. "On the K-Server
  Conjecture." *Journal of the ACM 1995.*
- Megiddo, N., Modha, D. S. "ARC: A Self-Tuning, Low Overhead
  Replacement Cache." *USENIX FAST 2003.*
- Towers, M., et al. "Gymnasium." 2023.
  https://gymnasium.farama.org/

---

## Appendix A: Reproduction

```bash
# Install
uv venv --python 3.12 .venv
.venv/bin/pip install -e .

# Sanity-check the reference baseline
.venv/bin/hlbench init --env pendulum --model reference-pd \
    --exp-id v0-paper-baseline
RUN_DIR=./runs/reference-pd/pendulum/v0-paper-baseline
cp agents/pd_pendulum/policy.py $RUN_DIR/workspace/system/policy.py
.venv/bin/hlbench serve --run-dir $RUN_DIR --env pendulum &
SERVER_PID=$!
sleep 1

# Run all 256 in-loop episodes in 32 batches
for i in $(seq 0 31); do
    .venv/bin/hlbench submit --env-instances $((i*8))-$((i*8+7))
done
.venv/bin/hlbench finalize
kill $SERVER_PID

# Inspect run.json
cat $RUN_DIR/run.json | jq '.outcome.final_score'
# Expect: 98.3 (within ±1.0 across re-runs due to variance)
```

For the v1 envs landed in v0.1:

```bash
# Hardcore pendulum (domain-randomized)
.venv/bin/hlbench init --env pendulum_hardcore --model my-agent \
    --exp-id pendulum-hardcore-trial-1

# Online algorithm: cache replacement
.venv/bin/hlbench init --env cache_replacement --model my-agent \
    --exp-id cache-trial-1

# (etc. for all 6 v1 envs landed)
```

## Appendix B: Verdict Enum

The complete verdict enum (10 verdicts in v0.1, after the
`oversize` deletion in this release):

| Verdict | Phase | Budget consumed |
|---|---|---|
| `ok` | 6→7 | yes (`N` requested) |
| `budget_invalid` | 1 | **no** (free retry) |
| `invalid_env_instance` | 1 | **no** (free retry) |
| `missing_policy` | 3 | yes (`N`) |
| `denied_import` | 3 | yes (`N`) |
| `import_error` | 4 | yes (`N`) |
| `init_timeout` | 5 | yes (`N`) |
| `init_error` | 5 | yes (`N`) |
| `oom` | 6 | yes (`N`); partial preserved |
| `submit_wall_exceeded` | 6 | yes (`N`); partial preserved |

See `docs/submit-protocol.md` for the full submission lifecycle.
