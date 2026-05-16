# Pendulum Swing-Up

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Apply a continuous torque action to swing the pendulum upright. This is a public
telemetry task: the policy and human task description both use
`[cos(theta), sin(theta), angular_velocity]`. The action is a one-dimensional
list containing torque in the environment range.
