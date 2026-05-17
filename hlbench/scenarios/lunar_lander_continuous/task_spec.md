# Lunar Lander Continuous

This file is scenario-author reference text. Generated workspaces receive
`task.md` and `task_contract.json`.

Land the spacecraft safely on the pad near the origin using continuous engine
controls. The policy receives the official LunarLander telemetry vector:
`[x_position, y_position, x_velocity, y_velocity, angle, angular_velocity,
left_leg_contact, right_leg_contact]`.

Return two continuous action values in `[-1, 1]`: `[main_engine_control,
lateral_booster_control]`. The main engine is off below `0` and scales from
50% to 100% throttle between `0` and `1`. The lateral booster is inactive near
`0`; values below `-0.5` or above `0.5` fire a side booster.
