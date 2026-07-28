"""Horizon-aware greedy baseline for Apple Incremental Game."""

from __future__ import annotations

import math
from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue

_MACHINE_IDS = 10
_LEVELS = 4


class BaselinePolicy:
    def __init__(self) -> None:
        self._capacities: tuple[int, ...] | None = None
        self._costs: tuple[tuple[int, ...], ...] | None = None

    def act(self, observation: PolicyValue) -> PolicyValue:
        state = _decode_observation(observation)
        initial = cast(
            tuple[
                tuple[int, ...],
                tuple[tuple[int, ...], ...],
            ]
            | None,
            state["initial"],
        )
        if initial is not None:
            self._capacities, self._costs = initial
        if self._capacities is None or self._costs is None:
            raise ValueError("initial machine configuration is missing")

        apples = cast(int, state["apples"])
        remaining = cast(int, state["turns_remaining"])
        machines = cast(tuple[tuple[int, ...], ...], state["machines"])
        powers = cast(tuple[tuple[int, ...], ...], state["powers"])

        if all(power == 0 for power in powers[0]):
            return {"upgrade": [0, 0]}

        best: tuple[int, int] | None = None
        best_benefit = 0
        best_cost = 1
        for level in range(_LEVELS):
            horizon_factor = (
                math.comb(remaining, level + 1)
                if remaining >= level + 1
                else 0
            )
            for machine_id in range(_MACHINE_IDS):
                cost = self._costs[level][machine_id] * (
                    powers[level][machine_id] + 1
                )
                if cost > apples or cost * 2 > apples:
                    continue
                lower_power = math.prod(
                    powers[lower_level][machine_id]
                    for lower_level in range(level)
                )
                benefit = (
                    self._capacities[machine_id]
                    * machines[level][machine_id]
                    * lower_power
                    * horizon_factor
                )
                if benefit <= cost:
                    continue
                if (
                    best is None
                    or benefit * best_cost > best_benefit * cost
                ):
                    best = (level, machine_id)
                    best_benefit = benefit
                    best_cost = cost
        if best is None:
            return None
        return {"upgrade": [best[0], best[1]]}


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()


def _decode_observation(
    observation: PolicyValue,
) -> dict[str, object]:
    if type(observation) is not dict or set(observation) != {
        "turn",
        "turns_remaining",
        "apples",
        "machines",
        "powers",
        "initial",
    }:
        raise ValueError("observation is invalid")
    turn = observation["turn"]
    remaining = observation["turns_remaining"]
    apples = observation["apples"]
    if (
        type(turn) is not int
        or type(remaining) is not int
        or type(apples) is not int
        or not 0 <= turn <= 500
        or remaining != 500 - turn
        or apples < 0
    ):
        raise ValueError("observation counters are invalid")
    machines = _matrix(observation["machines"], minimum=1)
    powers = _matrix(observation["powers"], minimum=0)
    initial = _initial(observation["initial"])
    return {
        "turn": turn,
        "turns_remaining": remaining,
        "apples": apples,
        "machines": machines,
        "powers": powers,
        "initial": initial,
    }


def _matrix(
    value: PolicyValue,
    *,
    minimum: int,
) -> tuple[tuple[int, ...], ...]:
    if (
        type(value) is not list
        or len(value) != _LEVELS
        or any(type(row) is not list or len(row) != _MACHINE_IDS for row in value)
    ):
        raise ValueError("machine matrix is invalid")
    result: list[tuple[int, ...]] = []
    for raw_row in value:
        row = cast(list[PolicyValue], raw_row)
        if any(type(item) is not int or item < minimum for item in row):
            raise ValueError("machine matrix is invalid")
        result.append(tuple(cast(list[int], row)))
    return tuple(result)


def _initial(
    value: PolicyValue,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]] | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"capacities", "costs"}:
        raise ValueError("initial configuration is invalid")
    raw_capacities = value["capacities"]
    if (
        type(raw_capacities) is not list
        or len(raw_capacities) != _MACHINE_IDS
        or any(type(item) is not int or item < 1 for item in raw_capacities)
    ):
        raise ValueError("capacities are invalid")
    costs = _matrix(value["costs"], minimum=1)
    return tuple(cast(list[int], raw_capacities)), costs
