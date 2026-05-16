# Mountain Car

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Drive the car up the right hill. This is a Gymnasium Classic Control telemetry
task: the policy receives the official MountainCar observation
`[position, velocity]`. Return `0` for left acceleration, `1` for no
acceleration, or `2` for right acceleration. The fixed track requires building
momentum by moving back and forth before the car can climb the right hill.
