"""Intentionally weak zero-torque Ant starting point."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    contact = context.environment_parameters.get(
        "include_cfrc_ext_in_observation"
    )
    if type(contact) is not bool:
        raise ValueError("include_cfrc_ext_in_observation is invalid")
    return BaselinePolicy()
