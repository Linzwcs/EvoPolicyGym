"""An intentionally weak no-op Stable-Retro baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return 0


def make_policy(context: PolicyContext) -> BaselinePolicy:
    game = context.environment_parameters.get("game")
    state = context.environment_parameters.get("state")
    if game != "Airstriker-Genesis-v0" or state != "Level1":
        raise ValueError("Airstriker environment parameters are invalid")
    return BaselinePolicy()
