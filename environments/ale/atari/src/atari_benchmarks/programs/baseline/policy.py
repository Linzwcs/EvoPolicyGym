"""An intentionally weak no-op ALE baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return 0


def make_policy(context: PolicyContext) -> BaselinePolicy:
    game = context.environment_parameters.get("game")
    if game != "Tetris":
        raise ValueError("game is invalid")
    return BaselinePolicy()
