# Gymnasium MuJoCo

MuJoCo environments provide seeded continuous-control tasks using Gymnasium's
official built-in models. Every leaf is independently installable and pins the
shared `gymnasium[mujoco]` dependency in its own lockfile.

Only current `v5` tasks are targeted. Deprecated `mujoco-py` registrations and
the superseded `v4` contracts are not separate Benchmarks.

| Environment | Status | Max steps | Observation | Action |
| --- | --- | ---: | --- | --- |
| [Reacher-v5](reacher/) | Implemented | 50 | 10 named arm/target values | 2 joint torques |
| [Pusher-v5](pusher/) | Implemented | 100 | 23 named arm, object, and goal values | 7 joint torques |
| [InvertedPendulum-v5](inverted_pendulum/) | Implemented | 1000 | 4 named cart/pole values | 1 force |
| [InvertedDoublePendulum-v5](inverted_double_pendulum/) | Implemented | 1000 | 9 named cart/two-pole values | 1 force |
| [Swimmer-v5](swimmer/) | Implemented | 1000 | 8 or 10 named body values | 2 joint torques |
| [Hopper-v5](hopper/) | Implemented | 1000 | 11 or 12 named body values | 3 joint torques |
| [Walker2d-v5](walker2d/) | Implemented | 1000 | 17 or 18 named body values | 6 joint torques |
| [HalfCheetah-v5](half_cheetah/) | Implemented | 1000 | 17 or 18 named body values | 6 joint torques |
| [Ant-v5](ant/) | Implemented | 1000 | 27/29 body values plus 78 nested contact values | 8 joint torques |
| [Humanoid-v5](humanoid/) | Implemented | 1000 | 45/47 state plus optional nested dynamics | 17 joint torques |
| [HumanoidStandup-v5](humanoid_standup/) | Implemented | 1000 | 45/47 state plus optional nested dynamics | 17 joint torques |

Official XML model selection is Host-owned and fixed to Gymnasium's packaged
asset for each Benchmark. Custom XML paths and rendering/camera parameters are
not Policy-visible environment parameters.

HalfCheetah additionally publishes bounded, lossless rendered-frame evidence
and an H.264 MP4 derived from the same step-aligned captures. The remaining
MuJoCo distributions currently retain their semantic state traces and are the
next visual-evidence rollout target.
