"""An intentionally weak neutral-action HighwayEnv baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def __init__(self, *, continuous: bool, action_size: int) -> None:
        self._continuous = continuous
        self._action_size = action_size

    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        if self._continuous:
            return [0.0] * self._action_size
        return 1


def make_policy(context: PolicyContext) -> BaselinePolicy:
    continuous = context.environment_parameters.get("continuous_actions")
    action_size = context.environment_parameters.get("action_size")
    if type(continuous) is not bool:
        raise ValueError("continuous_actions is invalid")
    if type(action_size) is not int or action_size <= 0:
        raise ValueError("action_size is invalid")
    return BaselinePolicy(continuous=continuous, action_size=action_size)

