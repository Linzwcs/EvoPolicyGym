"""Intentionally weak zero-torque HumanoidStandup starting point."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0] * 17


def make_policy(context: PolicyContext) -> BaselinePolicy:
    inertias = context.environment_parameters.get(
        "include_cinert_in_observation"
    )
    if type(inertias) is not bool:
        raise ValueError("include_cinert_in_observation is invalid")
    return BaselinePolicy()
