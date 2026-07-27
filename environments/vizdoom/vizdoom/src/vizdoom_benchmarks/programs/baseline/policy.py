"""An intentionally weak no-op ViZDoom baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def __init__(self, hybrid: bool) -> None:
        self._hybrid = hybrid

    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        if self._hybrid:
            return {"binary": 0, "continuous": [0.0, 0.0, 0.0]}
        return 0


def make_policy(context: PolicyContext) -> BaselinePolicy:
    hybrid = context.environment_parameters.get("hybrid_action")
    if type(hybrid) is not bool:
        raise ValueError("hybrid_action is invalid")
    return BaselinePolicy(hybrid)
