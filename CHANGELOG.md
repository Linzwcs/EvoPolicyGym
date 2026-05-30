# Changelog

All notable changes to hlbench-pro are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **OpenAI Codex CLI backend for `hlbench agent`** (alongside the
  existing Claude Code backend). Pick with `--backend {claude,codex}`
  on `hlbench agent` (default `claude`, byte-identical to prior
  behavior). Codex backend implementation in
  ``src/hlbench_harness/codex_agent.py`` (~360 lines) wraps
  ``codex exec --json`` (turn 0) / ``codex exec resume <id>``
  (turn 1+). Codex 0.133+ does not let callers pre-allocate a session
  UUID — the harness scrapes the id from the first ``session_meta``
  JSONL event on turn 0, then reuses it for every subsequent turn's
  ``exec resume``. If turn 0 emits no ``session_meta`` (binary
  missing, auth failure), turn 1 falls back to a fresh ``codex exec``
  with a logged warning rather than resuming against a bogus id.
  Per-turn artifacts (``turn_NNN.{stream.jsonl,json,txt,prompt.txt}``)
  follow the same shape as the Claude backend so analyst tools work
  unchanged. Cost / token usage stay ``None`` for the codex backend
  (Codex 0.133's ``--json`` doesn't surface them). 13 new tests
  covering session-id scrape, the resume command shape, the
  no-``session_meta`` fallback, the bypass-vs-sandbox-mode mutual
  exclusion, and the cost-stays-None contract.
- **`--codex-binary`, `--codex-sandbox-mode`,
  `--no-codex-bypass-approvals`, `--no-require-codex`** —
  codex-only flags on `hlbench agent`.
- **Backend-aware defaults** for shared flags: `--model` defaults to
  `sonnet` for claude / `gpt-5-codex` for codex; `--turn-timeout`
  defaults to 600 s for claude / 900 s for codex (codex's first-call
  latency on macOS commonly busts 600).
- **`model_slug` on `harness_runner.json` / `agent.jsonl`** is now
  backend-aware: was always ``claude:<model>``; now
  ``<backend>:<model>`` (e.g. ``codex:gpt-5-codex``).
- **`scripts/run_v1_paper_matrix_codex.sh`** — codex-backend twin of
  the existing claude launcher. Same 16-env v1 paper roster, same
  parallelism shape, same run-dir layout; only the agent driver
  differs. Defaults: ``MODEL=gpt-5-codex``, ``MODEL_SLUG=codex-auto``,
  ``EXP_ID=v1paper-codex-<model>-<timestamp>``. Preflight checks
  ``codex`` instead of ``claude`` and points users at ``codex login``
  / ``OPENAI_API_KEY`` for auth. Drives ``scripts/run_matrix.py
  --backend codex`` under the hood. Both launchers can run
  side-by-side under different ``MODEL_SLUG``s without clobbering.

- **Moonshot Kimi Code CLI as a third backend for `hlbench agent`**.
  Pick with `--backend kimi`. Implementation in
  ``src/hlbench_harness/kimi_agent.py`` (~430 lines) wraps
  ``kimi -p`` (turn 0) / ``kimi -S <session-id> -p`` (turn 1+).
  Like codex, kimi auto-generates session ids; the harness resolves
  the id via a three-tier fallback: (1) scrape from
  ``--output-format stream-json`` events, (2) read
  ``~/.kimi-code/session_index.jsonl`` filtered by ``workDir``, (3)
  fall back to ``kimi -C`` (continue most-recent for cwd) with a
  logged warning. Each hlbench run uses a unique workspace_dir, so
  the index-lookup tier is exact in practice. Public ``session_id``
  is a harness-minted UUID4 (stable label); the kimi-internal id
  (``session_<uuid>``) is held privately for the resume command.
  Cost / token usage stay None (kimi 0.6's stream-json doesn't
  surface them). 19 new tests covering session-id scrape, the
  index-fallback path, the continue-fallback path, the yolo flag,
  and the cost-stays-None contract.
- **`--kimi-binary`, `--no-kimi-yolo`, `--no-require-kimi`** —
  kimi-only flags on `hlbench agent`. Default model is `kimi-k2`,
  default `--turn-timeout` is 900 s.
- **`scripts/run_v1_paper_matrix_kimi.sh`** — kimi-backend twin of
  the claude / codex launchers. Same 16-env v1 paper roster, same
  parallelism shape, same run-dir layout. Defaults: ``MODEL=kimi-k2``,
  ``MODEL_SLUG=kimi-auto``, ``EXP_ID=v1paper-kimi-<model>-<timestamp>``.
  Drives ``scripts/run_matrix.py --backend kimi`` under the hood.
  All three launchers (claude / codex / kimi) can run side-by-side
  under different ``MODEL_SLUG``s — directly produces the
  Sonnet × Codex × Kimi columns for paper/table.md.

- **`hlbench_harness` package + `hlbench-agent` CLI** — automated
  evaluation driver that runs Claude Code through one full
  `init → submit → finalize` loop on any registered env, preserving
  the inner agent's conversation context across iterations via
  `claude --print --resume <session_id>`. Contrast with
  `../hlbench`'s per-epoch fresh-subprocess design: state lives in the
  Claude session, not just in workspace files. See
  [`docs/dogfood.md`](./docs/dogfood.md).
  - `hlbench_harness.prompts` — initial + continuation prompt
    composition; pure functions, golden-tested.
  - `hlbench_harness.state` — bridges live `Server.info()` and
    on-disk `submit_NNN/summary.json` into a single `TurnObservation`.
  - `hlbench_harness.claude_agent` — `subprocess`-based wrapper
    around `claude --print --output-format=json`. Pre-allocates the
    session UUID so we never have to scrape it from the first reply.
    Per-turn artifacts: `<run_dir>/logs/agent_turns/turn_NNN.{json,txt,prompt.txt}`.
  - `hlbench_harness.runner` — `HarnessRunner` loop. Termination
    priority: ``budget_exhausted`` (preferred, when ``remaining_budget``
    hits 0 after the most recent turn) → ``consecutive_failures`` (N
    back-to-back failed turns) → ``max_turns`` (safety cap) →
    ``agent_finalized`` (defensive — agent shouldn't be calling
    ``/finalize`` per the prompt). Harness always calls
    ``Server.finalize()`` after the loop so ``run.json`` is always
    written. Persists ``<run_dir>/logs/harness_runner.json`` for
    analyst tooling.
- **30 new tests** covering prompt composition, state observation,
  loop control (with in-process `FakeAgent`), and subprocess wrapping
  (with stub `claude` shell script). 119 → 150 tests; mypy strict +
  ruff still clean.
- **CI workflow** (`.github/workflows/ci.yml`) — runs `pytest -q` +
  `ruff check` + `mypy --strict` on every push to `main` and every
  PR. Python 3.12 only (matches `requires-python` in `pyproject`).
  Will dangle until a remote is configured; harmless until then.
- **Cost + usage tracking.** ``ClaudeAgent`` now parses ``total_cost_usd``,
  ``num_turns``, and ``usage`` from claude's ``--output-format=json``
  envelope. Surfaces on ``TurnResult``, ``TurnLogEntry``, and as
  ``total_cost_usd`` + ``total_usage`` aggregates on ``RunSummary``
  (persisted in ``logs/harness_runner.json``). The CLI summary block
  prints total cost + input/output tokens when present. Test stubs
  that omit these fields yield ``None`` (and ``$0.00`` aggregate)
  rather than crashing.
- **``agent.jsonl`` writer** (``hlbench_harness/agent_log.py``) —
  closes the ``output.md §6.2`` known-limitation. ``HarnessRunner``
  now emits ``agent_start`` (model + session_id at run begin),
  one ``completion`` per turn (turn_index + input_tokens +
  output_tokens + cost_usd + latency_ms), and ``agent_end`` (reason +
  totals) on exit. Writes append-only to
  ``<run_dir>/logs/agent.jsonl``. Disable via
  ``HarnessRunner(agent_log=AgentLog.disabled())``. Failed writes are
  swallowed — observability never breaks the run.
- **Per-env starter policy auto-staged into the workspace.**
  ``EnvDefinition.starter_policy_path`` (registered via
  ``register_env(..., starter_policy_path=...)``) is copied into
  ``workspace/system/policy.py`` by ``Server.__init__`` if no
  policy.py exists yet. Pendulum ships
  ``src/hlbench/envs/pendulum/starter_policy.py`` (zero-torque
  baseline) — the contract is unambiguous from turn 0. Re-init on an
  existing run preserves the agent's edits.
- **Pendulum TASK.md tightened with explicit ``act()`` contract.**
  Added Python-level interface block, explicit input/output type
  table (np.ndarray (3,) float32 → np.ndarray (1,) float32), worked
  single-step example, and a JSON dump of obs_space + action_space
  matching ``GET /info``. Removes the "what does ``act()`` actually
  return" guesswork that wasted tokens in earlier live runs.
- **Streaming claude output (``--output-format=stream-json``).**
  ``ClaudeAgent`` now uses Popen + a line-buffered reader thread,
  writing every event (assistant thinking blocks, tool_use,
  tool_result, system, result) to
  ``logs/agent_turns/turn_NNN.stream.jsonl`` in real time. When a
  turn times out the agent's full thought-process trace up to the
  kill is preserved on disk — fixing the "we lost the agent's
  reasoning" failure mode observed in a live test. The final
  ``type:"result"`` event is still extracted into ``turn_NNN.json``
  for the existing cost/text accessors.
- **Hardcore env trio landed (`pendulum_hardcore`, `lunar_hardcore`,
  `bipedal_hardcore`).** First three v1-roster envs (#14-16 per
  `docs/envs.md`) implemented as new env packages alongside the
  existing v0 envs (no in-place replacement; v0 envs remain for
  backward compat). Per-env mechanics:
  - **`pendulum_hardcore`**: gym.Wrapper around `Pendulum-v1` that
    reassigns `m`, `l`, `g` per `reset(seed=...)` from train (nominal)
    or held-out (disjoint OOD) ranges based on seed magnitude.
    Train: m∈[0.5,2.0], l∈[0.7,1.5], g∈[8.0,12.0]; held-out:
    m∈[2.0,3.5], l∈[1.5,2.2], g∈[4.0,8.0].
  - **`lunar_hardcore`**: gym.Wrapper around `LunarLanderContinuous-v3`
    with `enable_wind=True`, reassigning `wind_power` and
    `turbulence_power` per seed from train vs held-out OOD ranges.
  - **`bipedal_hardcore`**: factory targets Gymnasium's
    `BipedalWalkerHardcore-v3` directly — its built-in procedural
    terrain (stumps, ladders, pits) provides per-seed variation
    without a wrapper.
  - Seed pool convention: train seeds in `[0, 1_000_000)`, held-out
    in `[1_000_000, 2_000_000)`. The wrapper's seed-magnitude check
    dispatches to train vs OOD ranges. Bipedal doesn't use this
    (no wrapper).
  - `tests/test_hardcore_envs.py` (9 tests) verifies registration,
    per-seed sampler ranges, train/held-out disjointness, and seed
    pool split-by-floor invariant. Box2D in-process loading
    segfaults on macOS pytest, so lunar/bipedal factory + reset is
    NOT exercised in-process — those envs are validated end-to-end
    via Sandbox-spawned subprocess tests instead.
  - 177 → 186 tests; mypy strict + ruff still clean.
- **Online algorithm env trio landed (`cache_replacement`,
  `k_server`, `online_bipartite_matching`).** Three more v1-roster
  envs (#11-13 per `docs/envs.md`). All custom-built pure-Python
  envs (no gymnasium dependency for game logic) — pass duck-typed
  Gymnasium API (reset/step/spec/observation_space/action_space).
  Each ships with a train pool drawn from a "natural" distribution
  and a held-out pool drawn from an adversarial / structured
  distribution disjoint from train, so a greedy/textbook policy
  that wins train will demonstrably degrade on held-out:
  - **`cache_replacement`**: capacity-8 cache, 64 object IDs, trace
    length 500. Train: Zipfian (LRU-friendly). Held-out: scan-heavy
    (cycles a permutation, LRU-hostile). `act()` returns a slot
    index to evict on miss.
  - **`k_server`**: 3 servers on `[-1,1]²` plane, 200 requests per
    episode. Train: 2-Gaussian mix at `(±0.4, ±0.4)`. Held-out:
    4 corners with 75% weight on one corner, defeating greedy
    nearest-server. `act()` returns server index `[0, K)`.
  - **`online_bipartite_matching`**: 16 left vertices, 24 online
    arrivals. Train: random `G(N, M, p=0.25)`. Held-out: KVV-style
    adversarial structure (first M/2 arrivals only see left half;
    second half adds "honey trap" edges that punish greedy).
    `act()` returns left vertex `[0, N)` to match, or `N` to skip.
  - All three use the same seed-magnitude convention: train seeds
    `[0, 1_000_000)`, held-out `[1_000_000, 2_000_000)`. The env's
    trace/request/graph generator reads the seed magnitude to
    dispatch distribution selection.
  - `tests/test_online_algo_envs.py` (15 tests): registration,
    factory + step round-trip, determinism, train-vs-held-out
    distribution divergence (quantitative — e.g. unique-IDs-in-100
    for cache, quadrant-fraction for k_server, structural-edge
    presence for matching), seed pool split-by-floor invariant.
  - 186 → 201 tests; mypy strict + ruff still clean. These envs
    use `Discrete` action spaces — first non-Box action types in
    the roster.
- **9 new gym-category envs landed (MuJoCo trio + MiniGrid quartet +
  CarRacing-lite).** Expands the registry's category coverage to all
  four major Gymnasium categories (Classic Control, Box2D, MuJoCo,
  MiniGrid). Per-env mechanics:
  - **MuJoCo locomotion (4)**: `half_cheetah` (HalfCheetah-v5,
    17-D obs / 6-D action), `hopper` (Hopper-v5, 11/3),
    `walker2d` (Walker2d-v5, 17/6), `ant` (Ant-v5, 105/8).
    Direct Gymnasium wraps; per-seed variation = initial-state
    perturbation only (no domain randomization wrapper). Held-out
    generalization is "robustness to init-state distribution".
  - **MiniGrid POMDP navigation (4)**: `minigrid_doorkey`
    (DoorKey-16x16-v0), `minigrid_keycorridor` (KeyCorridorS6R3-v0),
    `minigrid_lavacrossing` (LavaCrossingS11N5-v0),
    `minigrid_obstructedmaze` (ObstructedMaze-2Dlhb-v0). Each wraps
    MiniGrid's Dict obs into a flat 148-D uint8 Box (7×7×3 image +
    direction). Mission text is static per env (in TASK.md). Action
    space is the standard 7-action MiniGrid set.
  - **CarRacing-lite (1)**: `car_racing` wraps Gymnasium's
    `CarRacing-v3` with a block-average downsample from 96×96×3 →
    16×16×3. The downsampled obs fits inline (~3 KB serialized,
    under the 10 KB cap), avoiding the `observations.npy` infra
    dependency. The full 96×96 variant (`car_racing_pixel`) is
    deferred until external-storage infra lands.
  - 3 of the 9 (MuJoCo) require the `mujoco` optional dep
    (`pip install -e .[mujoco]`); 4 (MiniGrid) require `minigrid`;
    1 (CarRacing) uses already-required `gymnasium[box2d]`. All
    are registered if their import succeeds; partial-install setups
    will skip the corresponding env factories.
  - `tests/test_gym_category_envs.py` (25 tests): registration,
    seed pool sanity, obs/action shape matches Gymnasium spec,
    wrapper presence verification (for envs whose factory can't
    safely run in pytest main process). Factory + reset for
    MuJoCo/MiniGrid/CarRacing are validated end-to-end via
    Sandbox-spawned subprocess tests (existing
    test_submit_handler.py pattern).
  - 201 → 226 tests; mypy strict + ruff still clean across 63
    source files. Total registry: **20 envs** (5 v0 + 6 v1-batch1 +
    9 v1-batch2).
- **`observations.npy` infrastructure landed (closes the SPEC §4.6
  known-limitation).** Implements the `obs_storage="external"`
  side-car mechanism for envs whose per-step observations are too
  large to serialize inline (typically pixel envs > 10 KB JSON).
  Changes:
  - `EpisodeRecord` gains an optional `observations: list[np.ndarray]
    | None` field, populated by `run_episode` when called with
    `record_obs=False` (the existing flag the sandbox already pipes
    based on `env_meta.obs_storage`).
  - `feedback.write_observations(path, obs_list)` writes the per-step
    obs as `(episode_length, *obs_shape)` numpy arrays to
    `observations.npy`. Empty obs (e.g. reset_error before any step)
    write a zero-row file so consumers can rely on its presence.
  - `SubmitHandler` calls `write_observations` after `write_trajectory`
    when `rec.observations is not None`. `trajectory.jsonl` carries
    `"obs": null` for every step in this mode (already implemented).
  - End-to-end verified by `tests/test_visual_envs.py::
    test_pendulum_from_pixels_e2e_writes_observations_npy` — drives
    a real submit through the Sandbox subprocess and asserts
    `(episode_length, 64, 64, 3) uint8` shape on the resulting
    `observations.npy`, plus matching length to `trajectory.jsonl`.
- **2 new full-resolution visual envs (`car_racing_pixel`,
  `pendulum_from_pixels`).** First envs to use the new external-obs
  infrastructure:
  - **`car_racing_pixel`**: full 96×96×3 CarRacing-v3 (vs. the
    16×16 downsampled `car_racing` lite variant). `obs_storage =
    "external"`. Expert baseline ~900 (full resolution
    headroom is bigger than the lite's ~500).
  - **`pendulum_from_pixels`**: Pendulum-v1 with rendered RGB obs
    (64×64×3 from a `render_mode="rgb_array"` Gymnasium env,
    block-averaged from native 500×500). Tests visual extraction
    of physics state — angle from one frame, angular velocity from
    a 2-frame history (cached in Policy instance state across
    `act()` calls).
  - 12 new tests in `tests/test_visual_envs.py` covering the
    infra and both envs, including the slow but authoritative
    end-to-end Sandbox-driven submit test.
  - 226 → 238 tests; mypy strict + ruff still clean. Total
    registry: **22 envs** (20 prior + car_racing_pixel +
    pendulum_from_pixels).

### Changed

- **Consumer packages relocated under `src/` (standard src layout).**
  ``hlbench_cli/`` and ``hlbench_harness/`` moved from repo root to
  ``src/hlbench_cli/`` and ``src/hlbench_harness/`` — they're now
  siblings of ``src/hlbench/`` rather than top-level repo entries.
  pyproject's ``packages.find.where`` simplifies to just ``["src"]``.
  Lib/consumer separation is preserved at the package-boundary level:
  ``hlbench`` is the library, the other two are consumers that import
  it. Import paths (``from hlbench_cli.main import ...``) are
  unchanged — setuptools' ``package-dir`` mapping resolves both.
- **Per-env seed pools moved to ``data/`` subdirectories.**
  ``src/hlbench/envs/pendulum/{train,heldout}.json`` →
  ``src/hlbench/envs/pendulum/data/{train,heldout}.json``. Separates
  frozen seed data from executable env code; future envs follow the
  same shape (``src/hlbench/envs/<env_id>/data/``). Wheel-install
  friendly via ``package-data`` glob (``data/*.json``).
- **Console scripts unified under ``hlbench``**: the standalone
  ``hlbench-agent`` console script is removed; its functionality moves
  to ``hlbench agent`` as a subcommand of the main CLI. Flag
  definitions still live in ``src/hlbench_harness/__main__.py`` (the
  single source of truth) — ``src/hlbench_cli/main.py`` mounts them
  via ``add_subparser_args()``. The ``python -m hlbench_harness ...``
  fallback continues to work for ad-hoc invocation without the
  console script. Discoverability win: ``hlbench --help`` now lists
  all six operations (``init`` / ``serve`` / ``info`` / ``submit`` /
  ``finalize`` / ``agent``). Old scripts calling ``hlbench-agent ...``
  must update to ``hlbench agent ...``.
- **`--permission-mode` → `--claude-permission-mode`,
  `--allowed-tools` → `--claude-allowed-tools`.** Old names kept as
  deprecated aliases (argparse multi-name accepted) for one release;
  existing shell scripts continue to work. None of
  `scripts/run_matrix.py`, `scripts/run_bench.sh`, or
  `scripts/run_v1_paper_matrix.sh` set these flags so the rename is
  non-breaking. Introduced alongside the codex backend so backend-only
  knobs (claude vs codex) are namespace-tagged.

### Design notes

- **Finalize is harness-only in automated mode.** The agent's prompt
  explicitly says NOT to call ``POST /finalize`` — finalization is the
  harness's responsibility, triggered when the agent's budget is
  spent. The ``/finalize`` HTTP endpoint stays on the server for
  human/scripted use (see ``hlbench`` CLI), but ``hlbench-agent``
  never instructs the inner Claude to call it.

### Removed

- **`system/` size limits dropped.** ``system_total_bytes`` and
  ``system_single_file_bytes`` removed from
  ``GET /info:resource_limits`` and from ``AGENTS.md §4.3`` /
  ``SPEC §1.1`` — agents may grow ``system/`` without a cap. The
  fields had been advisory-only in code while documents claimed
  50KB/25KB enforcement; the discrepancy is resolved by deleting
  rather than enforcing. Heuristic-style policies (search, pattern
  databases, eval-function tables) need the headroom.
- **`oversize` verdict dropped.** With no size cap, the verdict
  has no trigger. Verdict enum drops from 11 → 10. Also removed
  from the per-event ``category`` enum in ``error.txt``. Phase 2
  (Snapshot) is now infallible from the agent's perspective —
  validation moves entirely into Phase 3. ``snapshot_size_bytes``
  is still recorded in ``checkpoints/_meta.json`` as a diagnostic.

### Planning

- **`docs/envs.md` — v1 environment roster (16 envs).** New doc
  specifying the v1 evaluation suite: 16 envs across 6 categories
  (visual control, procedural visual, spatial reasoning, visual
  game, online algorithms, hardcore state-based control), with
  per-env description / role / current solutions / policy-synthesis
  testability / expected discrimination / implementation cost. Each
  env is chosen such that "policy synthesis is the bottleneck" and
  "held-out generalization matters" — the two design pressures of
  the suite. Includes migration plan from v0's 5 envs (3 evolve
  into hardcore variants, 2 retire from the scored suite) and
  implementation-prerequisite list (`observations.npy` writer,
  per-env `act_wall_ms` override, anti-cheat extension for vision
  libs). This is a planning artifact — no envs implemented yet.

## [0.1.0a1] — 2026-05-29

Audit-and-polish release. Closes most of the post-MVP "known
limitations" enumerated in 0.1.0a0 and tightens a few SPEC corners
that emerged when wiring the full pipeline. No env additions —
Pendulum-v1 remains the only registered env.

### Added

**Sandbox + enforcement:**
- **`denied_imports` enforced** (`AGENTS.md §3.2`). New
  `hlbench.core.sandbox.DENIED_IMPORTS` set + a `sys.meta_path`
  import finder installed in the child before `system/` joins
  `sys.path`. Submit lifecycle now produces a clean `denied_import`
  verdict per `SPEC §4.1`. Limitation: stdlib modules pre-imported by
  Python startup (`subprocess`, `urllib`, `socket`) bypass the finder
  via `sys.modules` cache — eviction breaks gymnasium, so they're
  documented and pinned in tests instead. Full network blocking is a
  future hardening pass.
- **`submit_wall_s` enforced** (`SPEC §4.1`). `SubmitConfig.submit_wall_s`
  (default 300 s) caps cumulative Phase 6 wall time; exceeding it
  aborts remaining episodes with `submit_wall_exceeded`. Partial
  episodes preserved (the only verdict pair where `errors.txt` and
  `episodes/` legitimately coexist, per `submit-protocol §3.3`).

**Per-submit artifacts:**
- **`checkpoints/submit_NNN/`** (`output.md §5`). Each submit's
  snapshot is copied alongside an `_meta.json` matching `§5.2`.
  Both successful and failed submits get checkpoints — failed ones
  too, per `§5.3` "the snapshot is preserved so the agent can see
  what they submitted". Snapshot copy filters `__pycache__/`,
  `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.git/`, and
  `*.pyc` per `AGENTS.md §3.3` exclude list.
- **`episodes/ep_<XXX>/stdout.txt` and `stderr.txt`** (`SPEC §4.5`).
  Sandbox redirects child stdio to `io.StringIO` buffers; per-episode
  swap on `episode_done`. `Policy.__init__` output folds into the
  first episode's capture (no swap between `init_done` and ep 0,
  matching the spec). UTF-8-boundary-aware 64 KB truncation with a
  `... [truncated at 64KB] ...` marker line.

**Error files:**
- **64 KB cap on `errors.txt` and per-episode `error.txt`**
  (`SPEC §4.4.5`). Files now accept multiple appended events;
  exceeding the cap drops further events and writes a single
  `category: "truncated"` sentinel. First event always succeeds in
  full (the failure that produced it is too important to drop).

**New endpoint:**
- **`GET /task`** returns the env's task description as raw
  `text/markdown`. `TASK.md` is no longer staged into the workspace
  (CLAUDE.md invariant 5 tightens "4 things" → "3 things"). The env
  package still ships `TASK.md` as the source; `Server.task_md_text()`
  reads it on demand.

**Logging:**
- **`logs/harness.log`** (`output.md §6.1`). New
  `hlbench.core.harness_log.HarnessLog` writer + 7 lifecycle event
  types (`run_start`, `submit_received`, `snapshot_taken`,
  `episode_start`/`episode_end`, `submit_completed`, `finalize_start`,
  `run_end`). `run.json:artifacts.logs_harness` now points at it.
  Real seed values never appear (verified by test).

### Changed

- **Canonical run layout** (`output.md §1`). `Server(env_id, ...)` now
  takes `runs_root` (required) instead of `workspace_dir` (gone).
  Server computes `run_dir = runs_root / model / env / exp_id`
  internally and exposes `srv.run_dir` / `srv.workspace_dir` /
  `srv.exp_id` as public properties. CLI follows: `hlbench init
  --runs-root ./runs --model M --exp-id E` (auto-generated exp_id if
  absent, format per `output.md §2.3`); `hlbench serve --run-dir
  RUN_DIR --env X`. All run artifacts (workspace, checkpoints, logs,
  run.json) live under the single canonical run dir.
- **`AGENT.md` → `AGENTS.md`** rename. JSON field
  `agent_md_hash` → `agents_md_hash`; Python constants likewise.
- **`SPEC §1` workspace layout** is now 3 entries (AGENTS.md,
  system/, feedback/) — TASK.md no longer staged.

### Removed (deliberate spec simplifications)

- **`system/.final_submit` mechanism** dropped. The "most recent
  successful submit becomes the final policy" default covers ~all
  use cases; agents wanting to go back can copy from
  `checkpoints/submit_NNN/` and re-submit. `SPEC §5.4` updated.
- **`Policy.on_episode_end` hook** dropped. Episode returns already
  reach the LLM via `feedback/.../summary.json:returns`; the
  within-submit hook only duplicated that channel and muddied
  "what the agent did" vs "what the policy learned" attribution.
  Per-episode error category `on_episode_end_error` also removed.
  Policy interface is now exactly `__init__` + `reset` + `act`.

### Tests

119 tests across 9 files (was 87 in 0.1.0a0). Net +32, covering all
new features above. mypy strict + ruff still clean.

### Known limitations (deferred)

- Network blocking (the next deeper hardening pass — needs socket
  monkey-patching or namespacing).
- RSS-poll OOM detection (`submit_peak_rss_bytes` is set via
  `RLIMIT_AS` best-effort; macOS often ignores it under Apple's
  malloc).
- `observations.npy` external-obs storage (Pendulum only needs
  inline; required when CarRacing / pixel Atari arrive).
- `video.mp4` per-episode render.
- `agent.jsonl` agent-harness activity log (`output.md §6.2`) — we
  don't ship an agent harness; the operator's agent script is
  expected to write it.
- Only Pendulum-v1 ships; HalfCheetah / CarRacing / Atari remain
  on the post-0.1.0a1 roadmap.

## [0.1.0a0] — 2026-05-29

### Added

This is the MVP — single env (Pendulum-v1), full submit-to-finalize
loop end-to-end. See [`docs/findings.md`](./docs/findings.md) for
the calibration sweep that closed out the 14-day plan in
[`docs/architecture.md`](./docs/architecture.md).

**Server library** (`src/hlbench/`):
- `core.Server` — per-run server class with `info()` / `submit()` /
  `finalize()` (SPEC §3 server interface, SPEC §1.1 `/info` schema).
- `core.SubmitHandler` — 7-phase submit lifecycle with the unified
  11-verdict enum (`docs/submit-protocol.md` §2-3).
- `core.Sandbox` — `multiprocessing(spawn)` child holding one
  `Policy` instance, `signal.setitimer(ITIMER_REAL)` for `act()`
  wall-time, parent-side `Pipe.poll(timeout)` for init wall-time.
- `core.env_runner` — single-episode runner with the SPEC §4.2
  trajectory schema; categorizes per-episode failures as
  `act_error` / `act_timeout` / `reset_error` / `on_episode_end_error`.
- `core.heldout` — held-out evaluation that runs all M held-out
  seeds against a snapshotted final policy (SPEC §6).
- `core.scoring` — normalized score, `clip(0, 1.2) * 100` final
  score (SPEC §5.2), AUC + episodes-to-Npct auxiliary metrics
  (SPEC §5.3, normalized interpretation per the Day 14 spec fix).
- `core.feedback` — atomic `summary.json` writer, per-episode
  `trajectory.jsonl` with NaN/Inf encoded as strings, submit/episode
  `errors.txt` writers (SPEC §4.1-4.4).
- `core.seed_manager` — env-instance ID → hidden real seed
  resolver; train and held-out pools loaded from static JSON
  alongside the env package.
- `envs.registry` — `EnvDefinition` dataclass; `register_env()` /
  `get_env()`; `public_env_meta()` strips server-internal fields
  (`expert_baseline`, `random_baseline`, seed paths) per CLAUDE.md
  invariant 2.
- `envs.pendulum` — Pendulum-v1 with 256 train + 256 held-out
  seeds (disjoint, generated from `master_seed=42`), `TASK.md`
  template, expert/random baselines.
- `http_server` — stdlib `http.server` HTTP wrapper exposing
  `GET /info`, `POST /submit`, `POST /finalize` (CLAUDE.md
  invariant 8).

**Consumers** (outside `src/hlbench/` per the lib/consumer
separation rule):
- `hlbench_cli/` — argparse CLI: `hlbench {init,serve,info,submit,
  finalize}`. Range / list / mixed env-instance specs (`0-7`,
  `0,2,5`, `0-3,7`).
- `agents/pd_pendulum/policy.py` — reference energy-shaping
  swing-up + PD stabilize controller. Drop into
  `workspace/system/policy.py` to submit it.
- `scripts/gen_seeds.py` — regenerates `train.json`/`heldout.json`
  from `master_seed`. Reproducible: re-running with `--master-seed
  42` yields the byte-identical files checked in.
- `scripts/calibration.py` — sweeps the budget and tabulates
  `final_score` + auxiliary metrics. Output drives
  `docs/findings.md`.

**Documentation**:
- `README.md` — pitch, quick start, document map.
- `AGENTS.md` — agent rules, sandbox, anti-hack (kept as-shipped).
- `SPEC.md` — wire contract, scoring, held-out details.
- `docs/quickstart.md` — 60-second walkthrough plus agent-loop stub.
- `docs/architecture.md` — original implementation plan with a
  status banner and as-built layout.
- `docs/output.md` — `runs/<...>/` layout.
- `docs/submit-protocol.md` — 7 phases × 11 verdicts.
- `docs/findings.md` — calibration analysis (Day 14): `final_score`
  is constant for stateless policies (PD on Pendulum hits 98.3 at
  every budget); `auc_in_loop` rises with budget because of the
  `(0, 0)` start anchor; `held_out_gap = +8` validates determinism.

**Tests**: 87 across 8 files. Coverage 84% (gaps are sandbox
child-process code behind the multiprocessing barrier, `cmd_serve`
which blocks forever, and the deferred `oom` partial-execute path).

### Spec changes (Day 14)

- **SPEC §5.3** `episodes_to_50pct` / `episodes_to_80pct`: was
  `mean_return >= 0.5 × expert`, which inverts for negative-reward
  envs (Pendulum has expert ≈ -150, so 0.5 × expert ≈ -75 is
  unreachably *better* than expert). Now defined as
  `normalized_score(mean_return) >= 0.5`. Implementation in
  `scoring.episodes_to_threshold` already used the normalized form;
  the spec wording now matches.

### Known limitations

- `denied_imports` enforcement: not yet enforced (post-MVP bundle).
- Network blocking: not yet enforced (post-MVP bundle).
- RSS-poll OOM detection: not implemented; `RLIMIT_AS` is set
  best-effort but Apple's malloc often ignores it on macOS.
- `submit_wall_s` deadline: not enforced.
- `stdout.txt` / `stderr.txt` per-episode capture: not implemented.
- `observations.npy` external-obs storage: not implemented (Pendulum
  only needs inline).
- 64KB error-file truncation cap: not implemented.
- Per-submit code snapshots in `checkpoints/`: not produced.
- `agent.jsonl` / `harness.log` / `env.log`: not produced.
- Only Pendulum-v1 ships; HalfCheetah / CarRacing / Atari are on
  the post-MVP roadmap.

[Unreleased]: https://example.com/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://example.com/compare/v0.1.0a0...v0.1.0a1
[0.1.0a0]: https://example.com/releases/tag/v0.1.0a0
