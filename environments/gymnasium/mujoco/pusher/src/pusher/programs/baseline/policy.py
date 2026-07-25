"""Intentionally weak zero-torque Pusher starting point."""

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def act(self, observation: PolicyValue) -> PolicyValue:
        del observation
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    frame_skip = context.environment_parameters.get("frame_skip")
    if type(frame_skip) is not int or frame_skip <= 0:
        raise ValueError("frame_skip environment parameter is invalid")
    return BaselinePolicy()
