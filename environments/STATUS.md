# Environment integration status

This ledger covers the planned environment expansion. “Integrated” means an
independently installable Benchmark distribution exists, uses only the public
authoring SPI, owns a lockfile and tests, and has passed a real upstream
`reset()`/`step()` smoke test. “Deferred” means the requested environment
requires a Kernel capability that the current single-Policy ABI intentionally
does not claim. “Unavailable” means upstream runtime or redistributable assets
are absent; metadata-only registrations are not presented as runnable profiles.

Profiles are public, typed Benchmark configuration selected by the Host before
a Run. A selected profile is fixed for that Run, visible to the Coding Agent
and Policy through `environment_parameters`, and contributes to the
environment digest. Episode seeds and private scenario identity are not
profiles and never cross the Policy boundary.

## Integrated

| Ecosystem | Coverage | Distribution layout |
| --- | --- | --- |
| Crafter | Canonical `CrafterReward-v1` RGB task with all 17 actions and all 22 achievements; official shifted-geometric scoring, legacy survival-gated scoring, and additive survival-development v3 scoring | [`crafter/crafter/`](crafter/crafter/) |
| Standard MiniGrid | 19 requested tasks across 18 families: Fetch, MultiRoom, DynamicObstacles, ObstructedMazeFull and ObstructedMazeDlhb, PutNear, RedBlueDoors, BlockedUnlockPickup, UnlockPickup, Unlock, LockedRoom, Crossing, LavaGap, DistShift, GoToObject, GoToDoor, FourRooms, Playground, and Empty | One leaf distribution per family under [`minigrid/minigrid/`](minigrid/minigrid/); registered sizes and difficulty variants are profiles |
| MiniGrid WFC | All 22 requested presets: MazeSimple, DungeonMazeScaled, RoomsFabric, ObstaclesBlackdots, ObstaclesAngular, ObstaclesHogs2, ObstaclesHogs3, MazeKnot, MazeWall, Maze, MazeSpirals, MazePaths, Mazelike, RoomsOffice, RoomsMagicOffice, Dungeon, DungeonRooms, DungeonLessRooms, DungeonSpirals, Skew2, SkewCave, and SkewLake | [`minigrid/minigrid/wfc/`](minigrid/minigrid/wfc/) |
| BabyAI | All 40 requested tasks: GoToRedBallGrey, GoToRedBall, GoToRedBallNoDists, GoToObj, GoToLocal, GoTo, GoToImpUnlock, GoToSeq, GoToRedBlueBall, GoToDoor, GoToObjDoor, Open, OpenRedDoor, OpenDoor, OpenTwoDoors, OpenDoorsOrder, Pickup, UnblockPickup, PickupLoc, PickupDist, PickupAbove, PutNextLocal, PutNext, Unlock, UnlockLocal, KeyInBox, UnlockPickup, BlockedUnlockPickup, UnlockToUnlock, ActionObjDoor, FindObj, KeyCorridor, OneRoom, MoveTwoAcross, Synth, SynthLoc, SynthSeq, MiniBossLevel, BossLevel, and BossLevelNoUnlock | [`minigrid/babyai/`](minigrid/babyai/) |
| HighwayEnv | Highway, Merge, Roundabout, Intersection, TwoWay, Exit, UTurn, Parking, Racetrack, and LaneKeeping | [`highway_env/highway_env/`](highway_env/highway_env/) |
| Gymnasium-Robotics | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace, PointMaze, AntMaze, four Adroit Hand tasks, HandReach, Block/Egg/Pen manipulation with base, boolean-touch, and continuous-touch profiles, and FrankaKitchen (21 profiles) | [`gymnasium_robotics/robotics/`](gymnasium_robotics/robotics/) |
| MetaWorld | All 50 MT1 `*-v3` tasks, MT10, MT50, and fixed custom MT collections | [`metaworld/metaworld/`](metaworld/metaworld/) |
| NLE / NetHack | `NetHackScore-v0` with the fixed Monk role, upstream task action set, deterministic episode seed triplets, complete reversible raw training evidence, Agent-owned visual analysis, and aggregate-only Validation/Assessment | [`nle/nethack/`](nle/nethack/) |
| ALE Atari | The wheel's redistributable Tetris ROM, RGB observations, and the minimal action set | [`ale/atari/`](ale/atari/) |
| ViZDoom | 12 standard scenarios whose configs and WADs ship in ViZDoom: Basic, Audio, Notifications, DeadlyCorridor, Deathmatch, DefendCenter, DefendLine, HealthGathering, HealthGatheringSupreme, MyWayHome, PredictPosition, and TakeCover | [`vizdoom/vizdoom/`](vizdoom/vizdoom/) |
| Stable-Retro | The wheel's redistributable Airstriker Level 1 ROM/state and restricted discrete controller actions | [`stable_retro/airstriker/`](stable_retro/airstriker/) |

## Deferred by an explicit Kernel boundary

| Requested environments | Required capability | Why they are not represented as single-agent profiles |
| --- | --- | --- |
| MaMuJoCo | Multi-agent lifecycle and joint-action ABI | One Policy action and one observation cannot faithfully encode independently addressed agents |
| MPE2: Simple, SimpleAdversary, SimpleCrypto, SimpleFormation, SimpleLine, SimplePush, SimpleReference, SimpleSpeakerListener, SimpleSpread, SimpleTag, SimpleWorldComm, CollectTreasure | Multi-agent lifecycle and joint-action ABI | Parallel agents, per-agent observations/actions, termination, and rewards need a first-class contract |
| PettingZoo Classic: TicTacToe, ConnectFour, Chess, Go, RockPaperScissors, LeducHoldem, TexasHoldem, TexasHoldemNoLimit, GinRummy, Hanabi | AEC/parallel multi-agent ABI | Flattening turn ownership into the current action ABI would hide invalid-action and lifecycle semantics |
| PettingZoo Butterfly/SISL: CooperativePong, KnightsArchersZombies, Pistonball, Multiwalker, Pursuit | Parallel multi-agent ABI | Joint actions and per-agent termination/reward semantics are required |
| MetaWorld ML1, ML10, ML45 | Trial abstraction | Meta-learning support/query phases must not be confused with ordinary independent Episodes |
| MiniWoB++ original, no-delay, transfer, email, flight, and hidden-test suites | Whole-Run virtualization and a browser execution profile | Browser processes, web assets, and hidden server state need Host-owned lifecycle and isolation |

## Unavailable in the current portable runtime

| Requested environments | Result of integration attempt |
| --- | --- |
| Procgen's 16 games: BigFish, BossFight, CaveFlyer, Chaser, Climber, CoinRun, Dodgeball, FruitBot, Heist, Jumper, Leaper, Maze, Miner, Ninja, Plunder, and StarPilot | The currently published `procgen2` artifacts do not support this repository's Python 3.12/macOS ARM runtime (the available wheel is CPython 3.11 Linux x86-64 and no source distribution is published). Revisit with a maintained runtime or Docker execution profile. |
| Remaining ALE registrations | ALE 0.12.0 registers 104 environments, but its current wheel supplies a directly runnable ROM only for Tetris. The other 103 registrations fail without separately obtained ROMs. |
| Commercial ViZDoom maps | ViZDoom does not distribute commercial Doom/Doom II IWADs. Metadata or a Host-local WAD path is not a portable Benchmark distribution. |
| Remaining Stable-Retro integrations | Stable-Retro 1.0.1 ships 1,030 integration definitions but a redistributable ROM only for Airstriker. The other integrations require externally supplied ROMs. |

Deferred and unavailable rows have been processed and intentionally skipped;
they are not compatibility promises. A future implementation should add the
missing Kernel capability or legal asset provisioning first, then introduce a
new independent distribution with its own conformance and real-environment
tests.
