# Environment distributions

Environment implementations are independently installable Benchmark
distributions. This directory is organized by upstream ecosystem and then by
upstream suite:

```text
environments/
├── gymnasium/
│   └── classic_control/
│       ├── cartpole/
│       ├── acrobot/
│       ├── mountain_car/
│       ├── mountain_car_continuous/
│       └── pendulum/
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
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-cartpole` | `cartpole` | `gymnasium/CartPole-v1/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-acrobot` | `acrobot` | `gymnasium/Acrobot-v1/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-mountain-car` | `mountain_car` | `gymnasium/MountainCar-v0/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-mountain-car-continuous` | `mountain_car_continuous` | `gymnasium/MountainCarContinuous-v0/mean-return-v1` |
| [Gymnasium / Classic Control](gymnasium/classic_control/) | `evopolicygym-benchmark-pendulum` | `pendulum` | `gymnasium/Pendulum-v1/mean-return-v1` |
| [Jackdaw / Balatro](jackdaw/) | `evopolicygym-benchmark-balatro` | `balatro` | `jackdaw/Balatro/red-deck-white-stake/run-score-v2` |

From the repository root, select one leaf project explicitly:

```console
uv sync --project environments/gymnasium/classic_control/cartpole --extra dev
uv build environments/gymnasium/classic_control/cartpole
```

Future Gymnasium suites such as Toy Text, Box2D, and MuJoCo belong beside
`classic_control/`, not beside individual environments.
