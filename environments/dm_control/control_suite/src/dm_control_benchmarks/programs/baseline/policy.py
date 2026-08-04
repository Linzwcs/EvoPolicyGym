"""An intentionally weak zero-action dm_control baseline."""

import math
import struct

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue


def float64_values(value: PolicyValue) -> tuple[float, ...]:
    """Decode one public float64 TensorValue without NumPy."""

    if type(value) is not TensorValue or value.dtype != "float64":
        raise ValueError("observation field must be a float64 TensorValue")
    expected_values = math.prod(value.shape)
    if len(value.data) != expected_values * 8:
        raise ValueError("observation TensorValue byte length is invalid")
    return tuple(item[0] for item in struct.iter_unpack("<d", value.data))


class BaselinePolicy:
    def __init__(self, action_size: int) -> None:
        self._action_size = action_size

    def act(self, observation: PolicyValue) -> PolicyValue:
        if type(observation) is not dict:
            raise ValueError("observation must contain named TensorValue fields")
        # Decode every field so the starting Program demonstrates the public
        # TensorValue ABI even though this intentionally weak policy ignores it.
        for value in observation.values():
            float64_values(value)
        return [0.0] * self._action_size


def make_policy(context: PolicyContext) -> BaselinePolicy:
    action_size = context.environment_parameters.get("action_size")
    if type(action_size) is not int or action_size <= 0:
        raise ValueError("action_size is invalid")
    return BaselinePolicy(action_size)
