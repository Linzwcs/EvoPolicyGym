# Environment Integration Roadmap

> Status: active implementation. This page defines how EvoPolicyGym should connect
> all practical Gymnasium and Gymnasium-compatible environments. It is a coverage
> roadmap, not an evaluation-subset selection plan.

## Goal

EvoPolicyGym should expose Gymnasium-style tasks through one stable contract:
`Env`, `World`, `Case`, `Task`, `Secret`, and `Caps`. The target is broad
coverage through reusable adapters, not one bespoke wrapper per environment.

"All environments" means every selected registry id can be discovered, launched
with external split files, executed under budget control, and checked for valid
artifacts. Heavy or visual environments may live behind optional dependencies and
non-default test gates.

## Integration Levels

| Level | Meaning | Required Evidence |
|---|---|---|
| L0 Catalogued | Environment family and dependency are known. | Registry entry and install note. |
| L1 Smoke | Reset, random action, step, and close work. | `check_env` smoke passes. |
| L2 Tasked | Agent-facing task is usable. | `task.md`, split files, action/observation schema. |
| L3 Scored | Scores are comparable across runs. | Hidden anchors, normalization, artifact checker coverage. |
| L4 Calibrated | Real agents receive useful feedback. | At least one live-agent run with nontrivial failure/improvement signal. |

The structured manifest lives in `evopolicygym.envs.manifest`. Use
`uv run evopolicygym check-envs` to report each manifest entry's target level,
dependency group, registration status, and smoke-check result. Add `--discover`
to merge installed-but-unintegrated upstream registry ids as L0 backlog entries.
For L2 and above, `check-envs` also enforces a task-document quality gate:
`Objective`, `Policy Interface`, `Observation`, `Action`, and `Reward` sections
must be present in the agent-facing `/task` text.

## P0: Common Gymnasium Adapter

Status: in progress. The generic adapter is available under
`evopolicygym.envs.gym`; P1 aliases have concrete task text, and P2 native
families are registered at smoke level when their optional dependencies are
installed.

Dynamic bulk registration is available for installed Gymnasium registry ids.
Curated aliases remain the default registry surface; pass `--bulk` in CLI tools
or `registry(bulk=True)` in Python to expose long names such as
`gymnasium/CartPole-v1` without overriding short aliases such as
`gym/cartpole`.

Use `evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1` for broad
smoke passes; isolation keeps native crashes and browser/runtime setup failures
scoped to individual environment rows, while `--jobs` keeps 900+ row scans
practical.

Bulk task text follows a layered description policy: curated `TaskDoc` first,
then family-level templates, then a generic Gymnasium fallback. This keeps the
900+ environment surface useful without hand-writing one task document per id.

Build `evopolicygym.envs.gym` as the shared bridge for Gymnasium-compatible APIs.

Deliverables:

- `GymWorld` around `gymnasium.make(...)`, `reset(seed=...)`, `step(action)`,
  render, and close.
- Bulk long-name registration for installed ids, guarded behind explicit
  opt-in so default commands stay lightweight.
- Space codecs for `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`, `Tuple`,
  `Dict`, and nested combinations.
- Observation policy: inline small JSON-safe values; write large arrays to
  per-episode `observations.npy` or `observations.npz`.
- Action policy: parse JSON actions, clip only when configured, and report
  invalid/clipped actions in feedback diagnostics.
- Split loader/generator for `train.json`, `valid.json`, and `heldout.json`.
- Family-level `task.md` templates with per-env overrides.
- Optional dependency handling so base EvoPolicyGym stays lightweight. Keep
  conflicting runtime families in separate extras: `env-jax` covers Gymnasium's
  JAX-backed `phys2d/*` and `tabular/*` ids, while `env-mario` covers
  MO-Gymnasium's Mario id and is not co-installable with `env-jax` on the
  current NumPy constraints.

Gate: one vector env and one discrete/dict env pass L2 using the same adapter.

## P1: Official Low-Dependency Gymnasium

Status: in progress. The curated Classic Control and Toy Text aliases are
registered through the generic adapter. Each alias renders concrete `/task`
text from its `GymSpec`, and seed-backed data splits can be generated with
configurable train/validation/held-out sizes via `evopolicygym data make`.

Complete official built-ins that do not require native simulator stacks.

| Family | Target Coverage | Notes |
|---|---|---|
| Classic Control | `CartPole`, `Pendulum`, `MountainCar`, `MountainCarContinuous`, `Acrobot` | Cheap control, CI-friendly, good first adapter coverage. |
| Toy Text | `Blackjack`, `CliffWalking`, `FrozenLake`, `Taxi` | Discrete planning, sparse reward, symbolic policies. |

Deliverables:

- Registry aliases such as `gym/cartpole`, `gym/pendulum`, `gym/taxi`.
- Default split generator based on seeds and env kwargs.
- Trivial random/baseline policies for smoke diagnostics.
- Unit tests for space conversion, action parsing, and split reproducibility.

Gate: every P1 env reaches L3; CI runs a short L1/L2 subset.

## P2: Official Native-Dependency Gymnasium

Status: in progress. Box2D and MuJoCo aliases are integrated through the
generic Gymnasium adapter as L1 smoke targets. `gym/racing` is promoted to L2
with concrete task text and external image observation artifacts. Missing native
dependencies are treated as skipped capabilities instead of registry failures.

| Family | Target Coverage | Notes |
|---|---|---|
| Box2D | `LunarLander`, `BipedalWalker`, `CarRacing` | Requires Box2D extras; `CarRacing` exercises image observations. |
| MuJoCo | `Ant`, `HalfCheetah`, `Hopper`, `Humanoid`, `HumanoidStandup`, `InvertedPendulum`, `InvertedDoublePendulum`, `Pusher`, `Reacher`, `Swimmer`, `Walker2d` | Continuous control; heavier rollout cost but mostly vector observations. |

Deliverables:

- Registry aliases such as `gym/lunar`, `gym/racing`, `gym/reacher5`, and
  `gym/halfcheetah5`.
- Optional extras or install groups per simulator family.
- Resource profiles: expected episode time, artifact size, and CI eligibility.
- Per-family scoring anchors rather than hand-tuned anchors for every env first.

Gate: each family has full L1 coverage and at least two L3 representatives.

## P3: Binary Observations And Rendered Media

Status: in progress. Feedback writing now externalizes large fixed-shape
observations to `observations.npy` or `observations.npz`, trajectory rows use
`obs: null` for whole-observation external storage, and the run checker validates
that external observation rows match `trajectory.jsonl`. Gymnasium image-shaped
`Box` observations are exposed as `Image` schemas with external storage.

Finish artifact support required by image-first and render-heavy tasks.

Canonical per-episode layout:

```text
feedback/submit_NNN/episodes/ep_XXX/
├── trajectory.jsonl
├── observations.npy        # or observations.npz
├── video.mp4               # optional preview
├── stdout.txt
└── stderr.txt
```

Rules:

- `trajectory.jsonl:obs` is `null` when `obs_storage == "external"`.
- Single-array observations use shape `[episode_length, *obs_shape]`.
- Dict observations use named arrays in `.npz` when shapes are fixed.
- Video is a human-facing preview, not the canonical replay source.
- Checker validates trajectory length, observation rows, and video metadata when
  present.

Remaining work:

- Partial nested-field external references for mixed dict observations.
- Video preview writing and metadata validation.
- Additional visual-env promotions after `gym/racing`, starting with Atari/ALE
  once registry discovery and ROM handling are stable.

Gate: minimal writer/checker support has passed a synthetic image artifact test
and a real `gym/racing` smoke run; full P3 remains open for video and partial
nested-field externalization.

## P4: Official Atari / ALE

After P3, connect the official Atari/ALE registry through the same adapter.

Coverage target:

- Discover all installed ALE registry ids instead of maintaining a hand-written
  ROM list.
- Provide curated aliases for common tasks such as `Pong`, `Breakout`,
  `SpaceInvaders`, `MsPacman`, `Seaquest`, and `Qbert`.
- Support frame skip, sticky actions, grayscale/resize wrappers, and episode
  life settings as explicit env kwargs recorded in `run.json`.

Gate: all installed ALE ids reach L1; a curated subset reaches L3 after artifact
volume and rollout cost are measured.

## P5: Gymnasium-Compatible Single-Agent Ecosystem

Add high-visibility compatible libraries through family adapters or thin spec
files. These should reuse P0 codecs unless the project has unusual semantics.

| Family | Target Coverage | Special Work |
|---|---|---|
| MiniGrid | `Empty`, `DoorKey`, `Unlock`, `KeyCorridor`, `FourRooms`, `MultiRoom`, `DynamicObstacles`, `LavaCrossing`, `Memory`, BabyAI-style tasks | Mission text and symbolic grid projection. |
| MiniWorld | room, maze, hallway, wall-gap, pickup/navigation tasks | RGB observations and optional compact-state wrappers. |
| HighwayEnv | `highway`, `merge`, `roundabout`, `parking`, `intersection`, `racetrack` | Scenario ids, safety diagnostics, multiple observation modes. |
| Gymnasium-Robotics | Fetch, Shadow Hand, maze/goal-conditioned tasks | Dict observations and goal-conditioned scoring. |
| MetaWorld | reach, push, pick-place, door, drawer, button, window, sweep, faucet, hammer, assembly families | MuJoCo dependency and task-family metadata. |
| Procgen | `coinrun`, `maze`, `heist`, `jumper`, `caveflyer`, `dodgeball`, `miner`, `starpilot` | Pixel pipeline and procedural seed splits. |
| Safety-Gymnasium | point/car/velocity/button/push/navigation families | Constraint metrics and safety score components. |
| PyFlyt | hover, waypoint, landing, racing tasks | Continuous control and physics-resource profiling. |
| MiniWoB++ | browser form/click/text-entry tasks | DOM/screenshot observation contract and browser lifecycle. |

Gate: each family reaches L1 before custom scoring work; L3 requires stable task
text, hidden anchors, and reproducible split generation.

## P6: Bridges, Multi-Agent, And Exotic Spaces

These are valuable but may require contract extensions beyond the current
single-agent scalar-score model.

| Area | Candidates | Required Extension |
|---|---|---|
| Shimmy bridges | DeepMind Control Suite, BSuite, OpenSpiel, DeepMind Lab, legacy Gym | Adapter metadata for source API and wrapper stack. |
| Multi-agent | MAgent2, MPE2, MOMAland, PettingZoo-style tasks | Policy interface for multiple actors and team/opponent scoring. |
| Multi-objective | MO-Gymnasium, MOMAland | Vector reward capture and scalarization policy. |
| Domain wrappers | SUMO-RL, trading envs, cybersecurity envs, robotics simulators | Family-specific reset config, secrets, and artifact schemas. |

Gate: do not force these into the base contract. Add explicit capability flags
first, then implement adapters only where the semantics are clear.

## Automation And Coverage Tracking

Maintain a generated coverage report under `experiment/` or `docs/envs/` that
records, per env id:

- package/version and optional dependency group;
- observation/action spaces and render modes;
- current integration level L0-L4;
- last smoke command, date, and failure reason if blocked;
- whether the env is eligible for CI, local smoke, or heavy/manual runs.

This report should be generated by tooling, not hand-maintained tables.

## Execution Order

1. Implement P0 generic adapter and codecs.
2. Move current dependency-free `cartpole` behavior onto the generic adapter.
3. Complete P1 Classic Control and Toy Text to L3.
4. Add P2 Box2D and MuJoCo L1 aliases with optional dependencies.
5. Add P3 binary observation support before promoting visual-heavy work to L2/L3.
6. Add P4 Atari/ALE through dynamic registry discovery.
7. Add P5 compatible libraries one family at a time.
8. Add P6 only after capability flags make the contract explicit.

## Design Rules

- Prefer spec-driven registration over one source package per env.
- Keep environment data outside package code; env directories provide examples,
  not formal benchmark cases.
- Record every env kwarg, wrapper, seed, and split source in `run.json`.
- Treat dependency failures as skipped capability, not framework failure.
- Keep default tests cheap; heavy simulators and visual tasks run in marked suites.
