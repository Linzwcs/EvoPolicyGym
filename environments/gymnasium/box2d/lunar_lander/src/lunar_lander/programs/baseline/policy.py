"""Intentionally weak no-thrust LunarLander starting point."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def __init__(self, *, continuous: bool) -> None:
        self._continuous = continuous

    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0, 0.0] if self._continuous else 0


def make_policy(context: PolicyContext) -> BaselinePolicy:
    continuous = context.environment_parameters.get("continuous")
    if type(continuous) is not bool:
        raise ValueError("continuous environment parameter is invalid")
    return BaselinePolicy(continuous=continuous)
