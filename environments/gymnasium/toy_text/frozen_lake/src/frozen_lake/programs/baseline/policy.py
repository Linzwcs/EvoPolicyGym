"""A small value-iteration baseline built only from public Policy context."""

from __future__ import annotations

from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue

_ACTIONS = (0, 1, 2, 3)


class BaselinePolicy:
    def __init__(self, actions: tuple[int, ...]) -> None:
        self._actions = actions

    def act(self, observation: PolicyValue) -> PolicyValue:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        state = observation.get("state")
        if type(state) is not int or not 0 <= state < len(self._actions):
            raise ValueError("observation state is invalid")
        return self._actions[state]


def make_policy(context: PolicyContext) -> BaselinePolicy:
    parameters = context.environment_parameters
    raw_layout = parameters.get("map")
    raw_slippery = parameters.get("is_slippery")
    raw_success_rate = parameters.get("success_rate")
    if (
        type(raw_layout) is not list
        or not raw_layout
        or any(type(row) is not str for row in raw_layout)
    ):
        raise ValueError("environment map is invalid")
    if type(raw_slippery) is not bool:
        raise ValueError("is_slippery is invalid")
    if type(raw_success_rate) is not float:
        raise ValueError("success_rate is invalid")
    layout = tuple(cast(list[str], raw_layout))
    return BaselinePolicy(
        _value_iteration(
            layout,
            is_slippery=raw_slippery,
            success_rate=raw_success_rate,
        )
    )


def _value_iteration(
    layout: tuple[str, ...],
    *,
    is_slippery: bool,
    success_rate: float,
) -> tuple[int, ...]:
    rows = len(layout)
    columns = len(layout[0])
    values = [0.0] * (rows * columns)
    discount = 0.99

    for _ in range(10_000):
        updated = list(values)
        for state in range(rows * columns):
            row, column = divmod(state, columns)
            if layout[row][column] in {"H", "G"}:
                continue
            updated[state] = max(
                _action_value(
                    layout,
                    values,
                    state,
                    action,
                    is_slippery=is_slippery,
                    success_rate=success_rate,
                    discount=discount,
                )
                for action in _ACTIONS
            )
        if (
            max(
                abs(after - before)
                for after, before in zip(updated, values, strict=True)
            )
            < 1e-12
        ):
            values = updated
            break
        values = updated

    return tuple(
        max(
            _ACTIONS,
            key=lambda action: _action_value(
                layout,
                values,
                state,
                action,
                is_slippery=is_slippery,
                success_rate=success_rate,
                discount=discount,
            ),
        )
        for state in range(rows * columns)
    )


def _action_value(
    layout: tuple[str, ...],
    values: list[float],
    state: int,
    action: int,
    *,
    is_slippery: bool,
    success_rate: float,
    discount: float,
) -> float:
    outcomes: tuple[tuple[float, int], ...]
    if not is_slippery:
        outcomes = ((1.0, action),)
    else:
        failure_rate = (1.0 - success_rate) / 2.0
        outcomes = (
            (failure_rate, (action - 1) % 4),
            (success_rate, action),
            (failure_rate, (action + 1) % 4),
        )

    total = 0.0
    for probability, actual_action in outcomes:
        next_state = _move(layout, state, actual_action)
        columns = len(layout[0])
        row, column = divmod(next_state, columns)
        tile = layout[row][column]
        reward = 1.0 if tile == "G" else 0.0
        continuation = 0.0 if tile in {"H", "G"} else values[next_state]
        total += probability * (reward + discount * continuation)
    return total


def _move(layout: tuple[str, ...], state: int, action: int) -> int:
    rows = len(layout)
    columns = len(layout[0])
    row, column = divmod(state, columns)
    if action == 0:
        column = max(column - 1, 0)
    elif action == 1:
        row = min(row + 1, rows - 1)
    elif action == 2:
        column = min(column + 1, columns - 1)
    else:
        row = max(row - 1, 0)
    return row * columns + column
