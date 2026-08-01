"""An intentionally weak first-valid-action Jumanji baseline."""

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue


class BaselinePolicy:
    def __init__(self, *, action_kind: str, num_values: tuple[int, ...]) -> None:
        self._action_kind = action_kind
        self._num_values = num_values

    def act(self, observation: PolicyValue) -> PolicyValue:
        mask = observation.get("action_mask") if type(observation) is dict else None
        if self._action_kind == "discrete":
            if type(mask) is TensorValue and mask.dtype == "bool":
                for index, valid in enumerate(mask.data):
                    if valid:
                        return index
            return 0
        if type(mask) is TensorValue and mask.dtype == "bool":
            if mask.shape == self._num_values:
                for flat_index, valid in enumerate(mask.data):
                    if valid:
                        return _unravel(flat_index, mask.shape)
            if (
                len(set(self._num_values)) == 1
                and mask.shape == (len(self._num_values), self._num_values[0])
            ):
                width = self._num_values[0]
                return [
                    _first(mask.data[index * width : (index + 1) * width])
                    for index in range(len(self._num_values))
                ]
        return [0] * len(self._num_values)


def _first(values: bytes) -> int:
    for index, valid in enumerate(values):
        if valid:
            return index
    return 0


def _unravel(flat_index: int, shape: tuple[int, ...]) -> PolicyValue:
    result: list[PolicyValue] = [0 for _ in shape]
    for index in range(len(shape) - 1, -1, -1):
        flat_index, coordinate = divmod(flat_index, shape[index])
        result[index] = coordinate
    return result


def make_policy(context: PolicyContext) -> BaselinePolicy:
    action_kind = context.environment_parameters.get("action_kind")
    raw_num_values = context.environment_parameters.get("action_num_values")
    if action_kind not in {"discrete", "multi_discrete"}:
        raise ValueError("action_kind is invalid")
    if (
        type(raw_num_values) is not list
        or not raw_num_values
        or any(type(item) is not int or item <= 0 for item in raw_num_values)
    ):
        raise ValueError("action_num_values is invalid")
    num_values: list[int] = []
    for item in raw_num_values:
        if type(item) is not int:
            raise ValueError("action_num_values is invalid")
        num_values.append(item)
    return BaselinePolicy(
        action_kind=action_kind,
        num_values=tuple(num_values),
    )
