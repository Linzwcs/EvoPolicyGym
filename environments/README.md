# Environment distributions

Environment implementations are independently installable Benchmark
distributions. This directory is organized by upstream ecosystem and then by
upstream suite:

```text
environments/
├── ale/
│   └── atari/
├── atcoder/
│   ├── ahc054/
│   │   └── treants_forest/
│   ├── ahc057/
│   │   └── molecules/
│   └── ahc058/
│       └── apple_incremental_game/
├── codechef/
│   └── june18/
│       └── warehouseman/
├── gymnasium/
│   ├── box2d/
│   ├── classic_control/
│   ├── mujoco/
│   └── toy_text/
├── gymnasium_robotics/
│   └── robotics/
├── highway_env/
│   └── highway_env/
├── jackdaw/
│   └── balatro/
├── metaworld/
│   └── metaworld/
├── minigrid/
│   ├── babyai/
│   └── minigrid/
│       ├── blocked_unlock_pickup/
│       ├── crossing/
│       ├── ... 18 additional standard families ...
│       └── wfc/
├── stable_retro/
│   └── airstriker/
└── vizdoom/
    └── vizdoom/
```

Only leaf directories are Python projects. Collection directories do not own a
`pyproject.toml`, lockfile, virtual environment, or combined dependency set.
Each leaf distribution owns its simulator dependency, static Benchmark
specification, deterministic Episode planning, Environment adapter, scoring,
public Feedback, baseline Program, lockfile, and tests.

The taxonomy describes source ownership and dependency provenance. It does not
change public Python imports, distribution names, or Benchmark IDs.

The [integration ledger](STATUS.md) records every planned environment,
including exact task/profile coverage and the environments deferred because a
multi-agent, Trial, browser, runtime, or redistributable-asset boundary is
still absent.

| Collection | Distribution | Import | Benchmark ID |
| --- | --- | --- | --- |
| [AtCoder / AHC054](atcoder/ahc054/) | `evopolicygym-benchmark-treants-forest` | `treants_forest` | `atcoder/AHC054/TreantsForest/capped-mean-turns-v1` |
| [AtCoder / AHC057](atcoder/ahc057/) | `evopolicygym-benchmark-molecules` | `molecules` | `atcoder/AHC057/Molecules/mean-log-cost-score-v1` |
| [AtCoder / AHC058](atcoder/ahc058/) | `evopolicygym-benchmark-apple-incremental-game` | `apple_incremental_game` | `atcoder/AHC058/AppleIncrementalGame/mean-log2-score-v1` |
| [CodeChef / June 2018](codechef/june18/) | `evopolicygym-benchmark-warehouseman` | `warehouseman` | `codechef/WAREHOUS/Warehouseman/mean-normalized-cost-v1` |
| [Gymnasium / Box2D](gymnasium/box2d/) | `evopolicygym-benchmark-bipedal-walker` | `bipedal_walker` | `gymnasium/BipedalWalker-v3/mean-return-v1` |
| [Gymnasium / Box2D](gymnasium/box2d/) | `evopolicygym-benchmark-car-racing` | `car_racing` | `gymnasium/CarRacing-v3/mean-return-v1` |
| [Gymnasium / Box2D](gymnasium/box2d/) | `evopolicygym-benchmark-lunar-lander` | `lunar_lander` | `gymnasium/LunarLander-v3/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-cartpole` | `cartpole` | `gymnasium/CartPole-v1/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-acrobot` | `acrobot` | `gymnasium/Acrobot-v1/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-mountain-car` | `mountain_car` | `gymnasium/MountainCar-v0/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-mountain-car-continuous` | `mountain_car_continuous` | `gymnasium/MountainCarContinuous-v0/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-pendulum` | `pendulum` | `gymnasium/Pendulum-v1/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-ant` | `ant` | `gymnasium/Ant-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-half-cheetah` | `half_cheetah` | `gymnasium/HalfCheetah-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-hopper` | `hopper` | `gymnasium/Hopper-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-humanoid` | `humanoid` | `gymnasium/Humanoid-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-humanoid-standup` | `humanoid_standup` | `gymnasium/HumanoidStandup-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-inverted-double-pendulum` | `inverted_double_pendulum` | `gymnasium/InvertedDoublePendulum-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-inverted-pendulum` | `inverted_pendulum` | `gymnasium/InvertedPendulum-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-pusher` | `pusher` | `gymnasium/Pusher-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-reacher` | `reacher` | `gymnasium/Reacher-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-swimmer` | `swimmer` | `gymnasium/Swimmer-v5/mean-return-v1` |
| [Gymnasium / MuJoCo](gymnasium/mujoco/) | `evopolicygym-benchmark-walker2d` | `walker2d` | `gymnasium/Walker2d-v5/mean-return-v1` |
| [Gymnasium / Toy Text](gymnasium/toy_text/) | `evopolicygym-benchmark-blackjack` | `blackjack` | `gymnasium/Blackjack-v1/mean-reward-v1` |
| [Gymnasium / Toy Text](gymnasium/toy_text/) | `evopolicygym-benchmark-cliff-walking` | `cliff_walking` | `gymnasium/CliffWalking-v1/mean-return-v1` |
| [Gymnasium / Toy Text](gymnasium/toy_text/) | `evopolicygym-benchmark-frozen-lake` | `frozen_lake` | `gymnasium/FrozenLake-v1/success-rate-v1` |
| [Gymnasium / Toy Text](gymnasium/toy_text/) | `evopolicygym-benchmark-taxi` | `taxi` | `gymnasium/Taxi-v4/mean-return-v1` |
| [Jackdaw / Balatro](jackdaw/) | `evopolicygym-benchmark-balatro` | `balatro` | `jackdaw/Balatro/red-deck-white-stake/run-score-v2` |
| [MiniGrid / MiniGrid](minigrid/minigrid/) | `evopolicygym-benchmark-minigrid-doorkey` | `minigrid_doorkey` | `minigrid/DoorKey-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/) | `evopolicygym-benchmark-minigrid-keycorridor` | `minigrid_keycorridor` | `minigrid/KeyCorridor-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/) | `evopolicygym-benchmark-minigrid-memory` | `minigrid_memory` | `minigrid/Memory-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/blocked_unlock_pickup/) | `evopolicygym-benchmark-minigrid-blocked-unlock-pickup` | `minigrid_blocked_unlock_pickup` | `minigrid/BlockedUnlockPickup-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/crossing/) | `evopolicygym-benchmark-minigrid-crossing` | `minigrid_crossing` | `minigrid/Crossing-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/dist_shift/) | `evopolicygym-benchmark-minigrid-dist-shift` | `minigrid_dist_shift` | `minigrid/DistShift-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/dynamic_obstacles/) | `evopolicygym-benchmark-minigrid-dynamic-obstacles` | `minigrid_dynamic_obstacles` | `minigrid/DynamicObstacles-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/empty/) | `evopolicygym-benchmark-minigrid-empty` | `minigrid_empty` | `minigrid/Empty-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/fetch/) | `evopolicygym-benchmark-minigrid-fetch` | `minigrid_fetch` | `minigrid/Fetch-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/four_rooms/) | `evopolicygym-benchmark-minigrid-four-rooms` | `minigrid_four_rooms` | `minigrid/FourRooms-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/go_to_door/) | `evopolicygym-benchmark-minigrid-go-to-door` | `minigrid_go_to_door` | `minigrid/GoToDoor-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/go_to_object/) | `evopolicygym-benchmark-minigrid-go-to-object` | `minigrid_go_to_object` | `minigrid/GoToObject-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/lava_gap/) | `evopolicygym-benchmark-minigrid-lava-gap` | `minigrid_lava_gap` | `minigrid/LavaGap-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/locked_room/) | `evopolicygym-benchmark-minigrid-locked-room` | `minigrid_locked_room` | `minigrid/LockedRoom-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/multiroom/) | `evopolicygym-benchmark-minigrid-multiroom` | `minigrid_multiroom` | `minigrid/MultiRoom-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/obstructed_maze/) | `evopolicygym-benchmark-minigrid-obstructed-maze` | `minigrid_obstructed_maze` | `minigrid/ObstructedMaze-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/playground/) | `evopolicygym-benchmark-minigrid-playground` | `minigrid_playground` | `minigrid/Playground-v0/room-coverage-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/put_near/) | `evopolicygym-benchmark-minigrid-put-near` | `minigrid_put_near` | `minigrid/PutNear-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/red_blue_doors/) | `evopolicygym-benchmark-minigrid-red-blue-doors` | `minigrid_red_blue_doors` | `minigrid/RedBlueDoors-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/unlock/) | `evopolicygym-benchmark-minigrid-unlock` | `minigrid_unlock` | `minigrid/Unlock-v0/success-rate-v1` |
| [MiniGrid / MiniGrid](minigrid/minigrid/unlock_pickup/) | `evopolicygym-benchmark-minigrid-unlock-pickup` | `minigrid_unlock_pickup` | `minigrid/UnlockPickup-v0/success-rate-v1` |
| [MiniGrid / WFC](minigrid/minigrid/wfc/) | `evopolicygym-benchmark-minigrid-wfc` | `minigrid_wfc` | `minigrid/WFC-v0/success-rate-v1` |
| [MiniGrid / BabyAI](minigrid/babyai/) | `evopolicygym-benchmark-minigrid-babyai` | `minigrid_babyai` | `minigrid/BabyAI-{family}-v0/success-rate-v1` |
| [HighwayEnv](highway_env/highway_env/) | `evopolicygym-benchmark-highway-env` | `highway_benchmarks` | `highway-env/{environment_id}/mean-return-v1` |
| [Gymnasium-Robotics](gymnasium_robotics/robotics/) | `evopolicygym-benchmark-gymnasium-robotics` | `robotics_benchmarks` | `gymnasium-robotics/{environment_id}/success-rate-v1` |
| [MetaWorld](metaworld/metaworld/) | `evopolicygym-benchmark-metaworld` | `metaworld_benchmarks` | `metaworld/{collection}/success-rate-v1` |
| [ALE Atari](ale/atari/) | `evopolicygym-benchmark-atari` | `atari_benchmarks` | `ale/Tetris-v5/mean-return-v1` |
| [ViZDoom](vizdoom/vizdoom/) | `evopolicygym-benchmark-vizdoom` | `vizdoom_benchmarks` | `vizdoom/{environment_id}/mean-return-v1` |
| [Stable-Retro](stable_retro/airstriker/) | `evopolicygym-benchmark-airstriker` | `airstriker` | `stable-retro/Airstriker-Genesis-v0/mean-score-delta-v1` |

From the repository root, select one leaf project explicitly:

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv build environments/gymnasium/classic_control/cartpole
```

Future Gymnasium suites belong beside `box2d/`, `classic_control/`, `mujoco/`,
and `toy_text/`, not beside individual environments.
