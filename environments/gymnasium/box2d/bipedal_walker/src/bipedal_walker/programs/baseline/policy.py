"""Intentionally weak zero-torque BipedalWalker starting point."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0, 0.0, 0.0, 0.0]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    hardcore = context.environment_parameters.get("hardcore")
    if type(hardcore) is not bool:
        raise ValueError("hardcore environment parameter is invalid")
    return BaselinePolicy()
