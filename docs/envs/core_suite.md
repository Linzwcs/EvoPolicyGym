# Core-16 Evaluation Suite

> Status: selected for paper experiments. This suite is intentionally smaller
> than EvoPolicyGym's broad L1 coverage surface. It is designed for controlled
> multi-agent experiments, not for exhaustive environment coverage.

## Selection Rule

Core-16 uses eight capability families with two environments per family. The
suite avoids browser dependencies so it can run on headless servers without
Playwright or Chromium, while still covering vector control, symbolic planning,
robotics, structured driving, and pixel observations.

All selected environments passed L1 smoke checks on 2026-06-04 with:

```bash
evopolicygym check-envs --bulk --isolate --jobs 4 --min-level L1 --timeout 60
```

## Environment List

| Capability family | Environment | Upstream id | What it tests |
|---|---|---|---|
| Classic hard control | `gym/acrobot` | `Acrobot-v1` | Nonlinear swing-up control, delayed reward, momentum-building strategies. |
| Classic hard control | `gym/continuouscar` | `MountainCarContinuous-v0` | Continuous throttle tuning with delayed goal reward and action-cost tradeoffs. |
| Box2D | `gym/bipedal` | `BipedalWalker-v3` | Contact-rich locomotion, balance, and long-horizon stability. |
| Box2D / pixel | `gym/racing` | `CarRacing-v3` | Pixel observation control and visual road-following policy design. |
| MuJoCo locomotion | `gym/halfcheetah5` | `HalfCheetah-v5` | High-dimensional continuous locomotion and speed optimization. |
| MuJoCo locomotion | `gym/ant5` | `Ant-v5` | Multi-legged coordination, stability, and high-dimensional action control. |
| MuJoCo manipulation | `gym/pusher5` | `Pusher-v5` | End-effector control, contact dynamics, and object-to-goal manipulation. |
| MuJoCo manipulation | `gym/reacher5` | `Reacher-v5` | Short-horizon arm control and geometric target reaching. |
| MiniGrid planning | `gymnasium/MiniGrid-DoorKey-16x16-v0` | `MiniGrid-DoorKey-16x16-v0` | Larger key-door subgoal decomposition with longer sparse-reward exploration under symbolic observations. |
| MiniGrid planning | `gymnasium/MiniGrid-KeyCorridorS4R3-v0` | `MiniGrid-KeyCorridorS4R3-v0` | Multi-room search, key retrieval, and object interaction. |
| MiniGrid navigation | `gymnasium/MiniGrid-FourRooms-v0` | `MiniGrid-FourRooms-v0` | Sparse-reward navigation through bottlenecks and partial observations. |
| MiniGrid navigation | `gymnasium/MiniGrid-ObstructedMaze-1Q-v1` | `MiniGrid-ObstructedMaze-1Q-v1` | Maze navigation with obstacles and interaction sequencing. |
| Highway driving | `gymnasium/parking-v0` | `parking-v0` | Goal-directed vehicle control and precise continuous maneuvering. |
| Highway driving | `gymnasium/roundabout-v0` | `roundabout-v0` | Structured traffic interaction, lane decisions, and collision avoidance. |
| Goal-conditioned robotics | `gymnasium/FetchPush-v4` | `FetchPush-v4` | Goal-conditioned object pushing with achieved/desired goal feedback. |
| Goal-conditioned robotics | `gymnasium/FetchPickAndPlace-v4` | `FetchPickAndPlace-v4` | Goal-conditioned grasping, lifting, and placement. |

## Not In Core

- `gym/cartpole` and `gym/pendulum` remain useful development smoke tests but
  are too simple for the main paper suite.
- BrowserGym MiniWoB++ is L1-ready, but excluded from Core-16 because the
  target server may not provide browser support. Treat it as an optional web
  extension.
- Atari/ALE is excluded from Core-16 for now. It is a good appendix candidate
  after pixel preprocessing, ROM handling, and artifact volume are calibrated.
- JAX and Mario are optional dependency families and are not part of the primary
  environment stack.

## Next Gates

Core-16 is currently selected and L1-smoked. Before final paper runs, promote
each environment through:

1. L2 task quality: verify agent-facing task text, observation/action schema,
   and deterministic split generation.
2. L3 scoring: set stable random/expert anchors or family-level normalization.
3. L4 calibration: run at least one live coding agent and verify feedback causes
   nontrivial policy changes.
