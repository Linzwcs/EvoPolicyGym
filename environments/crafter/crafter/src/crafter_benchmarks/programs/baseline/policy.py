"""Deterministic wander-and-interact Crafter starting Policy."""

import random

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

_MOVEMENT = (1, 2, 3, 4)
_CRAFTING = (8, 11, 12, 14, 15, 9, 13, 16, 10, 7)


class BaselinePolicy:
    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self._steps = 0

    def act(self, observation: PolicyValue) -> PolicyValue:
        if (
            type(observation) is not TensorValue
            or observation.dtype != "uint8"
            or observation.shape != (64, 64, 3)
        ):
            raise ValueError("Crafter observation is invalid")
        self._steps += 1
        if self._steps % 53 == 0:
            return _CRAFTING[(self._steps // 53) % len(_CRAFTING)]
        if self._steps % 29 == 0:
            return 6
        if self._steps % 3 == 0:
            return 5
        return self._random.choice(_MOVEMENT)


def make_policy(context: PolicyContext) -> BaselinePolicy:
    if context.environment_parameters.get("area") != [64, 64]:
        raise ValueError("Crafter area is invalid")
    if context.environment_parameters.get("image_size") != [64, 64]:
        raise ValueError("Crafter image size is invalid")
    return BaselinePolicy(context.policy_seed)
