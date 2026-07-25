"""Intentionally weak starting point for Acrobot development."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return 1


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()
