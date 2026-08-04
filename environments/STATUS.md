# Environment integration status

This ledger covers the planned environment expansion. “Integrated” means an
independently installable Benchmark distribution exists, uses only the public
authoring SPI, owns a lockfile and tests, and has passed either a real upstream
`reset()`/`step()` smoke test or rules-conformance tests for an independent
simulator. “Deferred” means the requested environment requires a Kernel
capability that the current single-Policy ABI intentionally does not claim.
“Unavailable” means upstream runtime or redistributable assets are absent;
metadata-only registrations are not presented as runnable profiles.

Profiles are public, typed Benchmark configuration selected by the Host before
a Run. A selected profile is fixed for that Run, visible to the Coding Agent
and Policy through `environment_parameters`, and contributes to the
environment digest. Episode seeds and private scenario identity are not
profiles and never cross the Policy boundary.

## Integrated

| Ecosystem | Coverage | Distribution layout |
| --- | --- | --- |
| ARC-AGI-3 | All 25 public games returned by the official discovery API on 2026-08-01, pinned by full version ID; custom fixed collections; fresh local game instances with per-Episode seeds supplied when supported upstream; one complete game per Episode with in-Episode level resets, shared official scorecard aggregation, and per-Episode playback GIF Feedback | [`arcprize/arc_agi_3/`](arcprize/arc_agi_3/) |
| AtCoder AHC054 | Treant's Forest with independently generated connected forests, private adventurer target orders, strict atomic Treant placement, and a 2,048-turn score cap | [`atcoder/ahc054/treants_forest/`](atcoder/ahc054/treants_forest/) |
| AtCoder AHC057 | Molecules with all 300 points, 1,000 turns, toroidal double-precision motion, atomic multi-bond Actions, momentum-conserving component velocity, exact 10 × 30 terminal partitions, and official logarithmic cost scoring | [`atcoder/ahc057/molecules/`](atcoder/ahc057/molecules/) |
| AtCoder AHC058 | Apple Incremental Game with the complete 10-ID, four-Level, 500-turn production hierarchy, independent log-uniform case generation, strict affordability, official processing order, and exact log2 scoring | [`atcoder/ahc058/apple_incremental_game/`](atcoder/ahc058/apple_incremental_game/) |
| CodeChef WAREHOUS | Warehouseman across the complete published 6–20 row and column range, with independent arrival generation, atomic full-solution validation, exact forklift dynamics, official normalized cost, and a guaranteed constructive baseline | [`codechef/june18/warehouseman/`](codechef/june18/warehouseman/) |
| Crafter | Canonical `CrafterReward-v1` RGB task with all 17 Actions and 22 achievements; official shifted-geometric scoring, long-horizon development, and additive survival-development scoring | [`crafter/crafter/`](crafter/crafter/) |
| DeepMind Control Suite | All 28 tasks in dm-control 1.0.43's official `suite.BENCHMARKING` collection across 14 domains; canonical named state observations, continuous rewards/actions, deterministic task randomization, per-Episode bounded default-camera replay GIF Feedback, and a MuJoCo 3.10 compatibility pin | [`dm_control/control_suite/`](dm_control/control_suite/) |
| Standard MiniGrid | 19 requested tasks across 18 families: Fetch, MultiRoom, DynamicObstacles, ObstructedMazeFull and ObstructedMazeDlhb, PutNear, RedBlueDoors, BlockedUnlockPickup, UnlockPickup, Unlock, LockedRoom, Crossing, LavaGap, DistShift, GoToObject, GoToDoor, FourRooms, Playground, and Empty | One leaf distribution per family under [`minigrid/minigrid/`](minigrid/minigrid/); registered sizes and difficulty variants are profiles |
| MiniGrid WFC | All 22 requested presets: MazeSimple, DungeonMazeScaled, RoomsFabric, ObstaclesBlackdots, ObstaclesAngular, ObstaclesHogs2, ObstaclesHogs3, MazeKnot, MazeWall, Maze, MazeSpirals, MazePaths, Mazelike, RoomsOffice, RoomsMagicOffice, Dungeon, DungeonRooms, DungeonLessRooms, DungeonSpirals, Skew2, SkewCave, and SkewLake | [`minigrid/minigrid/wfc/`](minigrid/minigrid/wfc/) |
| BabyAI | All 40 requested tasks: GoToRedBallGrey, GoToRedBall, GoToRedBallNoDists, GoToObj, GoToLocal, GoTo, GoToImpUnlock, GoToSeq, GoToRedBlueBall, GoToDoor, GoToObjDoor, Open, OpenRedDoor, OpenDoor, OpenTwoDoors, OpenDoorsOrder, Pickup, UnblockPickup, PickupLoc, PickupDist, PickupAbove, PutNextLocal, PutNext, Unlock, UnlockLocal, KeyInBox, UnlockPickup, BlockedUnlockPickup, UnlockToUnlock, ActionObjDoor, FindObj, KeyCorridor, OneRoom, MoveTwoAcross, Synth, SynthLoc, SynthSeq, MiniBossLevel, BossLevel, and BossLevelNoUnlock | [`minigrid/babyai/`](minigrid/babyai/) |
| HighwayEnv | Highway, Merge, Roundabout, Intersection, TwoWay, Exit, UTurn, Parking, Racetrack, and LaneKeeping | [`highway_env/highway_env/`](highway_env/highway_env/) |
| Jumanji | 18 single-Policy profiles: Game2048, GraphColoring, Minesweeper, RubiksCube and its partly-scrambled variant, Sudoku and its very-easy variant, SlidingTilePuzzle, BinPack, FlatPack, JobShop, Knapsack, Tetris, CVRP, Maze, Snake, TSP, and PacMan | [`jumanji/jumanji/`](jumanji/jumanji/) |
| Gymnasium-Robotics | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, PointMaze, AntMaze, four Adroit Hand tasks, HandReach, Block/Egg/Pen manipulation with base, boolean-touch, and continuous-touch profiles, and FrankaKitchen (21 profiles) | [`gymnasium_robotics/robotics/`](gymnasium_robotics/robotics/) |
| MetaWorld | All 50 MT1 `*-v3` tasks, MT10, MT50, and fixed custom MT collections | [`metaworld/metaworld/`](metaworld/metaworld/) |
| robosuite | All 19 environments registered by robosuite 1.5.2: Lift, Stack, five NutAssembly profiles, six PickPlace profiles, Door, Wipe, ToolHang, and four two-arm tasks; fixed Panda robots, BASIC/OSC pose control, state observations, per-Episode bounded `agentview` replay GIF Feedback, and a MuJoCo 3.3 compatibility pin | [`robosuite/robosuite/`](robosuite/robosuite/) |
| NLE / NetHack | Linux-targeted `NetHackScore-v0` with a fixed Monk role, upstream task Actions, deterministic Episode seed triplets, complete reversible raw training evidence, Agent-owned analysis, and aggregate-only Validation/Assessment | [`nle/nethack/`](nle/nethack/) |
| ALE Atari | The wheel's redistributable Tetris ROM, RGB observations, the minimal action set, and per-Episode bounded replay GIF Feedback | [`ale/atari/`](ale/atari/) |
| ViZDoom | 12 standard scenarios whose configs and WADs ship in ViZDoom: Basic, Audio, Notifications, DeadlyCorridor, Deathmatch, DefendCenter, DefendLine, HealthGathering, HealthGatheringSupreme, MyWayHome, PredictPosition, and TakeCover; per-Episode bounded replay GIF Feedback | [`vizdoom/vizdoom/`](vizdoom/vizdoom/) |
| Stable-Retro | The wheel's redistributable Airstriker Level 1 ROM/state, restricted discrete controller actions, and per-Episode bounded replay GIF Feedback | [`stable_retro/airstriker/`](stable_retro/airstriker/) |

## Deferred by an explicit Kernel boundary

| Requested environments | Required capability | Why they are not represented as single-agent profiles |
| --- | --- | --- |
| MaMuJoCo | Multi-agent lifecycle and joint-action ABI | One Policy action and one observation cannot faithfully encode independently addressed agents |
| MPE2: Simple, SimpleAdversary, SimpleCrypto, SimpleFormation, SimpleLine, SimplePush, SimpleReference, SimpleSpeakerListener, SimpleSpread, SimpleTag, SimpleWorldComm, CollectTreasure | Multi-agent lifecycle and joint-action ABI | Parallel agents, per-agent observations/actions, termination, and rewards need a first-class contract |
| PettingZoo Classic: TicTacToe, ConnectFour, Chess, Go, RockPaperScissors, LeducHoldem, TexasHoldem, TexasHoldemNoLimit, GinRummy, Hanabi | AEC/parallel multi-agent ABI | Flattening turn ownership into the current action ABI would hide invalid-action and lifecycle semantics |
| PettingZoo Butterfly/SISL: CooperativePong, KnightsArchersZombies, Pistonball, Multiwalker, Pursuit | Parallel multi-agent ABI | Joint actions and per-agent termination/reward semantics are required |
| Jumanji Cleaner, Connector, MMST, MultiCVRP, RobotWarehouse, LevelBasedForaging, and SearchAndRescue | Multi-agent lifecycle and joint-action ABI | Their observations, actions, and rewards address multiple agents; flattening them into one Policy action would erase agent ownership |
| MetaWorld ML1, ML10, ML45 | Trial abstraction | Meta-learning support/query phases must not be confused with ordinary independent Episodes |
| MiniWoB++ original, no-delay, transfer, email, flight, and hidden-test suites | Whole-Run virtualization and a browser execution profile | Browser processes, web assets, and hidden server state need Host-owned lifecycle and isolation |

## Unavailable in the current portable runtime

| Requested environments | Result of integration attempt |
| --- | --- |
| Procgen's 16 games: BigFish, BossFight, CaveFlyer, Chaser, Climber, CoinRun, Dodgeball, FruitBot, Heist, Jumper, Leaper, Maze, Miner, Ninja, Plunder, and StarPilot | The currently published `procgen2` artifacts do not support this repository's Python 3.12/macOS ARM runtime (the available wheel is CPython 3.11 Linux x86-64 and no source distribution is published). Revisit with a maintained runtime or Docker execution profile. |
| Remaining ALE registrations | ALE 0.12.0 registers 104 environments, but its current wheel supplies a directly runnable ROM only for Tetris. The other 103 registrations fail without separately obtained ROMs. |
| Commercial ViZDoom maps | ViZDoom does not distribute commercial Doom/Doom II IWADs. Metadata or a Host-local WAD path is not a portable Benchmark distribution. |
| Remaining Stable-Retro integrations | Stable-Retro 1.0.1 ships 1,030 integration definitions but a redistributable ROM only for Airstriker. The other integrations require externally supplied ROMs. |
| Jumanji Sokoban | Jumanji 1.1.1 creates its default generator by downloading the Boxoban dataset from Hugging Face. It is excluded until the distribution can pin and provision that dataset without network I/O during Environment construction. |
| ManiSkill 3.0.1 | The wheel registers 74 tasks and 61 declare no additional asset downloads, but no task is runnable in this native macOS ARM Host. Both `state`/`physx_cpu` construction and the documented `render_backend="none"` path still force SAPIEN's CPU renderer; ManiSkill's Darwin `can_render()` path unconditionally creates a Vulkan `RenderSystem`, which fails here with `ErrorIncompatibleDriver` because no Vulkan/MoltenVK driver is available. No Benchmark distribution is published until a Vulkan-capable Host profile is provisioned or upstream supports a truly renderer-free CPU scene. |

## Audited future candidates

These candidates have authoritative upstreams, but are not runnable
distributions in the current native release. Each row states the concrete
condition that must change before integration.

| Candidate | Current resolution |
| --- | --- |
| [Solomon VRPTW](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/) | Deferred. The authoritative page does not state redistribution terms for the instance files, and a completed route file is not yet an accepted constructive Policy interaction. EdgeBench instances, hidden cases, and best-known score anchors are not substitutes for either requirement. |
| [Battle for Wesnoth](https://github.com/wesnoth/wesnoth) | Deferred to a pinned headless engine profile that owns the executable, Lua AI bridge, independently authored scenarios, opponent policy, timeouts, and cleanup. A Python metadata package cannot make the external game process portable. |
| [OpenTTD](https://github.com/OpenTTD/OpenTTD) | Deferred to a pinned engine profile that owns the executable, base graphics, NoAI bridge, generated-map profile, long-running lifecycle, and deterministic teardown. |
| [Dungeon Crawl Stone Soup](https://github.com/crawl/crawl) | Deferred to a pinned headless engine profile and a verified supported machine-control surface. The EdgeBench Lua scaffold is not an upstream DCSS interface contract. |
| [OpenRCT2](https://github.com/OpenRCT2/OpenRCT2) | Deferred even after an engine profile exists because ordinary play additionally depends on separately licensed RollerCoaster Tycoon 2 data files. |
| [AtCoder AHC056](https://atcoder.jp/contests/ahc056/tasks/ahc056_a) and [CodeChef TRICOL](https://www.codechef.com/problems/TRICOL) | Not admitted to the current roadmap. Their authoritative form grades a completed design, and no meaningful incremental Policy interaction has been accepted. |
| EdgeBench-only graph, permutation, SMT, and fixed text-adventure leads | Rejected until each has an independently verified authoritative upstream, redistribution rights, and a non-artificial Environment contract. Fixed commercial text worlds additionally need publisher-level distribution rights. |

Deferred and unavailable rows have been processed and intentionally skipped;
they are not compatibility promises. A future implementation should add the
missing Kernel capability or legal asset provisioning first, then introduce a
new independent distribution with its own conformance and real-environment
tests.
