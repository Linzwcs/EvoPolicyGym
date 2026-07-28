"""One fresh independent Warehouseman Environment per Episode."""

from __future__ import annotations

from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .simulation import (
    MAX_INSTRUCTION_CHARACTERS,
    InvalidInstruction,
    WarehouseCase,
    WarehouseSimulation,
    generate_case,
    normalized_cost,
)


class WarehousemanEnvironment:
    """Atomic adapter for one complete public WAREHOUS solution."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Warehouseman does not use an Episode scenario")
        self._seed = episode.environment_seed
        self._case: WarehouseCase | None = None
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        self._case = generate_case(self._seed)
        self._started = True
        return {
            "rows": self._case.rows,
            "columns": self._case.columns,
            "arrivals": list(self._case.arrivals),
            "instruction_limit": MAX_INSTRUCTION_CHARACTERS,
        }

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started or self._case is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")
        if type(action) is not str:
            raise InvalidAction()

        simulation = WarehouseSimulation(self._case)
        try:
            simulation.execute(action)
        except InvalidInstruction:
            raise InvalidAction() from None

        self._done = True
        cost = normalized_cost(
            len(action),
            self._case.rows,
            self._case.columns,
        )
        return Step(
            observation={"status": "completed"},
            reward=cost,
            terminated=True,
            metrics={
                "normalized_cost": cost,
                "instruction_characters": len(action),
                "moves": simulation.moves,
                "picks": simulation.picks,
                "drops": simulation.drops,
                "loads": simulation.loads,
                "unloads": simulation.unloads,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._case = None
        self._closed = True


__all__ = ["WarehousemanEnvironment"]
