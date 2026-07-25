# Gymnasium collection

This collection contains independently distributed EvoPolicyGym Benchmarks
backed by Gymnasium environments. Subdirectories follow Gymnasium's upstream
suite organization.

| Suite | Status | Distributions |
| --- | --- | --- |
| [Classic Control](classic_control/) | Implemented | CartPole, Acrobot, Mountain Car, Continuous Mountain Car, Pendulum |
| Toy Text | Planned | Added as individual leaf distributions |
| Box2D | Planned | Added as individual leaf distributions with suite-local dependencies |
| MuJoCo | Planned | Added as individual leaf distributions with suite-local dependencies |

The collection itself is not installable. Select a leaf distribution and use
its own `pyproject.toml` and `uv.lock`.
