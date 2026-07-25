"""Intentionally weak zero-torque HalfCheetah starting point."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    excluded = context.environment_parameters.get(
        "exclude_current_positions_from_observation"
    )
    if type(excluded) is not bool:
        raise ValueError(
            "exclude_current_positions_from_observation is invalid"
        )
    return BaselinePolicy()
