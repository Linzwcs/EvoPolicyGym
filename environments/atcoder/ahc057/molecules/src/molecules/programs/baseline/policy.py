"""Guaranteed complete immediate-grouping baseline for Molecules."""

from typing import cast

from evopolicygym.policy import PolicyContext, PolicyValue


class BaselinePolicy:
    def __init__(self) -> None:
        self._first_turn = True

    def act(self, observation: PolicyValue) -> PolicyValue:
        if type(observation) is not dict:
            raise ValueError("observation must be an object")
        turn = observation.get("turn")
        if type(turn) is not int:
            raise ValueError("observation turn is invalid")
        if self._first_turn:
            if turn != 0:
                raise ValueError("first observation must be turn zero")
            self._first_turn = False
            bonds = [
                [group_start, point]
                for group_start in range(0, 300, 30)
                for point in range(group_start + 1, group_start + 30)
            ]
            return {"bonds": cast(list[PolicyValue], bonds)}
        return {"bonds": []}


def make_policy(context: PolicyContext) -> BaselinePolicy:
    del context
    return BaselinePolicy()
