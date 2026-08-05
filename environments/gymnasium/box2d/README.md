# Gymnasium Box2D

Box2D environments provide seeded physics tasks with richer dynamics and
heavier suite-local dependencies than Classic Control or Toy Text.

| Environment | Package | Policy observation | Policy action |
| --- | --- | --- | --- |
| [LunarLander](lunar_lander/) | `evopolicygym-benchmark-lunar-lander` | Eight named physical state values | Four discrete engines or two continuous controls |
| [BipedalWalker](bipedal_walker/) | `evopolicygym-benchmark-bipedal-walker` | Named body, joint, contact, and lidar state | Four continuous motor controls |
| [CarRacing](car_racing/) | `evopolicygym-benchmark-car-racing` | Exact `uint8[96,96,3]` RGB Tensor | Three continuous controls or five discrete Actions |

LunarLander and BipedalWalker preserve bounded upstream RGB captures
losslessly and derive H.264 MP4 videos from the same step-aligned frames.
CarRacing already preserves its native Policy-visible RGB observations and
publishes a derived replay GIF.

Each leaf distribution owns its Gymnasium `box2d` dependency, lockfile,
configuration, adapter, scoring, public trace, baseline Program, and tests.
