"""Intentionally weak starting point for CartPole development."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return 0


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()
