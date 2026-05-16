# Bipedal Walker

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Control a four-joint walker so it moves forward across slightly uneven terrain
without falling. The policy receives the official BipedalWalker state vector:
hull motion, joint positions and speeds, leg contact flags, and 10 lidar
rangefinder measurements. The observation does not include absolute
coordinates.

Return four continuous motor commands in `[-1, 1]` for the two hips and two
knees.
