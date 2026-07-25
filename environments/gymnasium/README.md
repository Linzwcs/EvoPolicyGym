# Gymnasium collection

This collection contains independently distributed EvoPolicyGym Benchmarks
backed by Gymnasium environments. Subdirectories follow Gymnasium's upstream
suite organization. Together they cover all 23 current tasks in Gymnasium's
four documented built-in suites.

| Suite | Status | Distributions |
| --- | --- | --- |
| [Box2D](box2d/) | Implemented | LunarLander, BipedalWalker, CarRacing |
| [Classic Control](classic_control/) | Implemented | CartPole, Acrobot, Mountain Car, Continuous Mountain Car, Pendulum |
| [MuJoCo](mujoco/) | Implemented | All eleven current `v5` tasks |
| [Toy Text](toy_text/) | Implemented | Blackjack, CliffWalking, FrozenLake, Taxi |

The collection itself is not installable. Select a leaf distribution and use
its own `pyproject.toml` and `uv.lock`. Alternate registrations for the same
task are configuration modes rather than duplicate distributions:
`FrozenLake8x8-v1`, `CliffWalkingSlippery-v1`,
`LunarLanderContinuous-v3`, and `BipedalWalkerHardcore-v3` are covered by the
corresponding typed Benchmark configuration. Deprecated versions, compatibility
shims, and Gymnasium's experimental `phys2d/` and `tabular/` implementations
are not separate Benchmarks.
