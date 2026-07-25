# Gymnasium Classic Control

The five Gymnasium Classic Control tasks are packaged as independent
EvoPolicyGym Benchmark distributions. They share an upstream suite and
Gymnasium dependency, but they do not import one another or share a Python
environment.

| Environment | Package | Policy observation | Policy action |
| --- | --- | --- | --- |
| [CartPole](cartpole/) | `evopolicygym-benchmark-cartpole` | Four named finite floats | Integer `0` or `1` |
| [Acrobot](acrobot/) | `evopolicygym-benchmark-acrobot` | Six named finite floats | Integer `0`, `1`, or `2` |
| [Mountain Car](mountain_car/) | `evopolicygym-benchmark-mountain-car` | Named position and velocity | Integer `0`, `1`, or `2` |
| [Continuous Mountain Car](mountain_car_continuous/) | `evopolicygym-benchmark-mountain-car-continuous` | Named position and velocity | Float in `[-1.0, 1.0]` |
| [Pendulum](pendulum/) | `evopolicygym-benchmark-pendulum` | Named angle coordinates and angular velocity | Float in `[-2.0, 2.0]` |

Each adapter converts Gymnasium's positional array into a bounded semantic
dictionary before the value crosses the Policy boundary. Public traces retain
the same observation value.
