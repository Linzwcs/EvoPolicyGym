"""Simple deterministic starting Policy for ARC-AGI-3."""

from __future__ import annotations

from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        state = observation.get("state")
        actions = observation.get("available_actions")
        if (
            type(state) is not str
            or type(actions) is not list
            or any(type(item) is not int for item in actions)
        ):
            raise ValueError("observation is invalid")
        if state == "GAME_OVER":
            return {"action": 0}
        available = cast(list[int], actions)
        if not available:
            return {"action": 0}
        action = available[0]
        if action == 6:
            return {"action": 6, "x": 32, "y": 32}
        return {"action": action}


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()


__all__ = ["BaselinePolicy", "make_policy"]
