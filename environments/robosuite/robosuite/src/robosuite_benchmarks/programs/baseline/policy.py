"""An intentionally weak zero-action robosuite baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def __init__(self, action_size: int) -> None:
        self._action_size = action_size

    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0] * self._action_size


def make_policy(context: PolicyContext) -> BaselinePolicy:
    action_size = context.environment_parameters.get("action_size")
    if type(action_size) is not int or action_size <= 0:
        raise ValueError("action_size is invalid")
    return BaselinePolicy(action_size)
