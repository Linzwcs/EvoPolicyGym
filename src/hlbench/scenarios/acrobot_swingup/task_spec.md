# Acrobot Swing-Up

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Swing the two-link acrobot so the free end reaches the target height. This is a
public telemetry task: the policy and human task description both use
`[cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta_dot1, theta_dot2]`.
Return `0` for negative torque, `1` for no torque, or `2` for positive torque.
