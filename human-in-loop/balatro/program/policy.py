"""EvoPolicyGym Policy ABI entrypoint."""

from policy_system.strategy import BalatroPolicy

from evopolicygym.policy import PolicyContext


def make_policy(context: PolicyContext) -> BalatroPolicy:
    del context
    return BalatroPolicy()
