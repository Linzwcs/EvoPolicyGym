# Environment distributions

Environment implementations are independently installable Benchmark
distributions. This directory is organized by upstream ecosystem and then by
upstream suite:

```text
environments/
├── gymnasium/
│   ├── box2d/
│   │   ├── bipedal_walker/
│   │   ├── car_racing/
│   │   └── lunar_lander/
│   ├── classic_control/
│   │   ├── cartpole/
│   │   ├── acrobot/
│   │   ├── mountain_car/
│   │   ├── mountain_car_continuous/
│   │   └── pendulum/
│   ├── mujoco/
│   │   ├── ant/
│   │   ├── half_cheetah/
│   │   ├── hopper/
│   │   ├── humanoid/
│   │   ├── humanoid_standup/
│   │   ├── inverted_double_pendulum/
│   │   ├── inverted_pendulum/
│   │   ├── pusher/
│   │   ├── reacher/
│   │   ├── swimmer/
│   │   └── walker2d/
│   └── toy_text/
│       ├── blackjack/
│       ├── cliff_walking/
│       ├── frozen_lake/
│       └── taxi/
└── jackdaw/
    └── balatro/
```

Only leaf directories are Python projects. Collection directories do not own a
`pyproject.toml`, lockfile, virtual environment, or combined dependency set.
Each leaf distribution owns its simulator dependency, static Benchmark
specification, deterministic Episode planning, Environment adapter, scoring,
public Feedback, baseline Program, lockfile, and tests.

The taxonomy describes source ownership and dependency provenance. It does not
change public Python imports, distribution names, or Benchmark IDs.

| Collection | Distribution | Import | Benchmark ID |
| --- | --- | --- | --- |
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

From the repository root, select one leaf project explicitly:

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv build environments/gymnasium/classic_control/cartpole
```

Future Gymnasium suites belong beside `box2d/`, `classic_control/`, `mujoco/`,
and `toy_text/`, not beside individual environments.
