# Environment Overview

> Status: planning note. This page organizes high-visibility Gymnasium and
> Gymnasium-compatible environments that could be adapted through EvoPolicyGym's
> v2 `Env` / `World` boundary.

For staged implementation work, see [`roadmap.md`](./roadmap.md). For the
current installed registry checklist, see [`env_list.md`](./env_list.md). The
machine-readable discovery output is [`discovered.json`](./discovered.json).

## Scope

EvoPolicyGym can connect environments that expose a Gymnasium-like episode loop:
`reset(...) -> observation` and `step(action) -> observation, reward,
terminated, truncated, info`. The adapter hides library details behind
`World.reset(case: Case)`, `World.step(action) -> Turn`, and `World.sample()`.

Formal runs should use external case split files, not package-internal seeds.
Each environment should load `train.json`, `valid.json`, and `heldout.json`,
then use `Case.data` for seeds, dynamics parameters, map ids, scenario ids,
task ids, or dataset rows.

## Runtime Assets

Some optional environment families need non-Python assets after `uv sync
with the relevant extra:

- Atari/ALE needs ROM files. Install them with `uv run AutoROM --accept-license
  --install-dir .venv/lib/python3.12/site-packages/ale_py/roms`.
- Gymnasium JAX registry ids such as `phys2d/CartPole-v1`,
  `phys2d/Pendulum-v0`, `tabular/Blackjack-v0`, and
  `tabular/CliffWalking-v0` need `uv sync --extra env-jax`.
- MO-Gymnasium `mo-supermario-v0` needs `uv sync --extra env-mario`.
  Keep this runtime separate from `env-jax`: the current Mario dependency chain
  requires `numpy<2.0`, while Gymnasium's JAX extra requires `numpy>=2.1`.
- MiniGrid WFC default presets use official pattern PNGs vendored under
  `src/evopolicygym/envs/gym/assets/minigrid_wfc_patterns/` when the installed
  `minigrid` wheel is missing those package data files.
- BrowserGym MiniWoB++ needs Playwright Chromium: `uv run python -m playwright
  install chromium`.
- BrowserGym MiniWoB++ also needs MiniWoB HTML files. Clone
  `Farama-Foundation/miniwob-plusplus` into `third_party/miniwob-plusplus`, use
  commit `7fd85d71a4b60325c6585396ec4f48377d049838`. EvoPolicyGym auto-detects
  this local path; alternatively set
  `MINIWOB_URL=file://<repo>/third_party/miniwob-plusplus/miniwob/html/miniwob/`.

## Built-In Gymnasium Families

These are the official Gymnasium environment families. Versioned registry ids
such as `CartPole-v1`, `LunarLander-v3`, or `HalfCheetah-v5` are variants of
the base tasks below.

| Family | Well-Known Tasks | Observation Style | EvoPolicyGym Use |
|---|---|---|---|
| Classic Control | `Acrobot`, `CartPole`, `MountainCar`, `MountainCarContinuous`, `Pendulum` | Low-dimensional vectors | First real control tasks; cheap and deterministic. |
| Toy Text | `Blackjack`, `CliffWalking`, `FrozenLake`, `Taxi` | Discrete/tabular state | Good for sparse reward, planning, and state-machine policies. |
| Box2D | `BipedalWalker`, `CarRacing`, `LunarLander` | Vectors or pixels depending on task | Stronger control tasks; native dependency risk. |
| MuJoCo | `Ant`, `HalfCheetah`, `Hopper`, `Humanoid`, `HumanoidStandup`, `InvertedDoublePendulum`, `InvertedPendulum`, `Pusher`, `Reacher`, `Swimmer`, `Walker2d` | Continuous vectors | Strong benchmark value; heavier rollout and dependency cost. |
| Atari / ALE | `Breakout`, `Pong`, `SpaceInvaders`, `MsPacman`, plus many ROM tasks | Pixel frames | Defer until image/video artifacts are first-class. |

Recommended built-in order: `Pendulum`, `MountainCar`, `FrozenLake` or `Taxi`,
then `LunarLander`. Keep `CartPole` as a smoke test only.

## Farama / Gymnasium-Compatible Ecosystem

These projects either follow Gymnasium APIs directly, are maintained in the
Farama ecosystem, or are commonly exposed through Gymnasium-compatible wrappers.

| Project | Representative Tasks | Observation Style | Fit |
|---|---|---|---|
| MiniGrid | `Empty`, `DoorKey`, `Unlock`, `KeyCorridor`, `FourRooms`, `MultiRoom`, `DynamicObstacles`, `LavaCrossing`, `Memory`, BabyAI-style instruction tasks | Symbolic grid tensor plus mission text | Excellent. Tests rule discovery, exploration, memory, and policy structure. |
| MiniWorld | `OneRoom`, `Maze`, `FourRooms`, `Hallway`, `WallGap`, pickup/navigation variants | 3D RGB or compact state via wrappers | Good later for navigation; visual burden is higher. |
| HighwayEnv | `highway`, `merge`, `roundabout`, `parking`, `intersection`, `racetrack` | Kinematics, occupancy grid, or grayscale image | Excellent. Structured scenarios, hidden traffic parameters, safety metrics. |
| Gymnasium-Robotics | `FetchReach`, `FetchPush`, `FetchSlide`, `FetchPickAndPlace`, Shadow Hand manipulation, maze/goal tasks | Goal-conditioned state dicts | Good after low-cost control; tests goal-conditioned feedback. |
| MetaWorld | reach, push, pick-place, door, drawer, button, window, sweep, faucet, hammer, assembly families | Continuous state vectors | Strong manipulation coverage; many tasks but MuJoCo-heavy. |
| MiniWoB++ | web tasks such as click, enter-text, choose-list, email/invoice/form variants | DOM fields plus screenshot | Very relevant for coding agents, but browser/Selenium heavy. |
| MO-Gymnasium | multi-objective variants of classic control, grid, mountain-car, lunar-lander-like tasks | Vector rewards | Useful for future multi-objective feedback studies. |
| MAgent2 / MPE2 | pursuit, battle, adversary, spread, speaker-listener style tasks | Multi-agent observations | Defer unless v2 adds multi-agent policy contracts. |
| MOMAland | multi-objective multi-agent tasks | Multi-agent, vector rewards | Defer; outside current single-agent scoring model. |
| Shimmy adapters | DeepMind Control Suite, OpenSpiel, BSuite, DeepMind Lab, Melting Pot, legacy Gym tasks | Wrapper-dependent | Useful bridge, but should be used after native envs are stable. |

## Other High-Visibility Compatible or Wrappable Envs

These are not all equally mature for EvoPolicyGym, but they are widely known in
RL and may be valuable once the core suite is stable.

| Domain | Projects / Tasks | Observation Style | Notes |
|---|---|---|---|
| Safety RL | `Safety-Gymnasium`, safety navigation, velocity, button, push tasks | Vectors, lidar-like state, sometimes images | Good for constraints and auxiliary metrics; scoring needs safety components. |
| Procedural Generalization | `Procgen` / `Procgen2`, coinrun, maze, jumper, caveflyer, dodgeball, heist, miner, starpilot | Pixels | Good generalization signal; defer until visual pipeline is ready. |
| Drone / Flight | `PyFlyt`, quadrotor fixed-wing hover/waypoint/racing tasks | Continuous vectors, optional render | Attractive control domain; dependency and physics cost moderate. |
| Driving / Traffic | `SUMO-RL`, `racecar-gym`, `BlueSky-Gym`, `Flow`-style traffic tasks | Structured scenario state or images | Good hidden-scenario benchmark; setup can be heavy. |
| Finance / Trading | `FinRL`, `gym-anytrading`, `gym-trading-env` | Time-series vectors | Useful for adaptation over regimes; high risk of benchmark leakage/overfitting. |
| Cybersecurity | `CyberBattleSim`, `NASim`, `Security-Gym`-style envs | Graph/state vectors | Strong coding-agent relevance; API normalization required. |
| Robotics Sim | `ManiSkill`, `robosuite`, Isaac-style Gym wrappers | State vectors or RGB-D | High-value long-term; heavy dependencies and simulator variance. |
| Biology / Medical | sepsis or ICU treatment simulators | Tabular/time-series state | Interesting decision tasks; require careful ethical/data framing. |

## Image Input Classification

| Category | Envs | Recommendation |
|---|---|---|
| Non-image by default | Classic Control, Toy Text, most MuJoCo, MetaWorld, Fetch/Hand robotics state tasks | Best first wave. Feedback can be textual JSON trajectories. |
| Symbolic image/tensor | MiniGrid | Good early. Convert grid tensor into object grid text or compact JSON. |
| Optional image | HighwayEnv, Box2D render modes, many robotics sims | Start with structured state; add image artifacts later. |
| Image-first | Atari/ALE, Procgen, MiniWorld, ViZDoom, CarRacing, MiniWoB++ screenshot channel | Defer until observation/video artifact support is stronger. |

## Integration Priority List

The coverage target is broad, but implementation should still unlock capability
classes in a deliberate order. Start with environments that create a clear gap
between one-shot coding and iterative improvement from feedback.

1. `Pendulum`: continuous action, cheap, controller tuning signal.
2. `MountainCar` or `MountainCarContinuous`: delayed reward and exploration.
3. MiniGrid `DoorKey` / `Unlock` / `FourRooms`: rule discovery and state
   machine construction.
4. `Taxi` or `FrozenLake`: tabular planning baseline with sparse feedback.
5. HighwayEnv `merge` / `parking`: structured scenario adaptation and safety.
6. `LunarLander`: stronger control benchmark once Box2D dependency is accepted.
7. MetaWorld reach/push/button subset: manipulation tasks after MuJoCo support
   is stable.

## Adapter Requirements

Each environment should provide:

1. `Env` registration with `Task`, `Secret`, `make`, optional `value`, `Caps`,
   and task text loaded from `task.md`.
2. A `World` adapter that converts observations and actions into JSON-safe
   `Turn` records.
3. External split files: `train.json`, `valid.json`, and `heldout.json`.
4. Normalization anchors (`random`, `expert`) hidden from the agent.
5. Small smoke sizes for tests and larger formal split sizes for experiments.

## Integration Rule

Prefer environments where feedback changes the code an agent should write:
controllers should be tuned, exploration should improve, state abstractions
should become cleaner, or hidden parameters should be handled more robustly.

If an agent solves the task perfectly on the first submit, the environment is
useful for smoke testing but weak for EvoPolicyGym's main research question.
