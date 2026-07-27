"""An intentionally weak zero-action MetaWorld baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0, 0.0, 0.0, 0.0]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    action_size = context.environment_parameters.get("action_size")
    if action_size != 4:
        raise ValueError("action_size is invalid")
    return BaselinePolicy()

