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
        shipment_count = len(self._case.arrivals)
        handling_characters = (
            simulation.picks + simulation.drops + 2 * (simulation.loads + simulation.unloads)
        )
        if simulation.moves + handling_characters != len(action):
            raise RuntimeError("Warehouseman instruction accounting drifted")
        if (
            simulation.picks != shipment_count
            or simulation.drops != shipment_count
            or simulation.loads < shipment_count
            or simulation.unloads < shipment_count
            or simulation.loads != simulation.unloads
        ):
            raise RuntimeError("Warehouseman completion accounting drifted")
        relocation_cycles = simulation.loads - shipment_count
        minimum_handling_characters = 6 * shipment_count
        excess_characters = len(action) - minimum_handling_characters
        if excess_characters != simulation.moves + 4 * relocation_cycles:
            raise RuntimeError("Warehouseman overhead accounting drifted")
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
                "rows": self._case.rows,
                "columns": self._case.columns,
                "warehouse_cells": self._case.rows * self._case.columns,
                "shipments": shipment_count,
                "instruction_characters": len(action),
                "instruction_characters_remaining": (MAX_INSTRUCTION_CHARACTERS - len(action)),
                "instruction_budget_fraction": (len(action) / MAX_INSTRUCTION_CHARACTERS),
                "characters_per_shipment": (len(action) / shipment_count),
                "moves": simulation.moves,
                "moves_per_shipment": simulation.moves / shipment_count,
                "picks": simulation.picks,
                "drops": simulation.drops,
                "loads": simulation.loads,
                "unloads": simulation.unloads,
                "total_operations": (
                    simulation.moves
                    + simulation.picks
                    + simulation.drops
                    + simulation.loads
                    + simulation.unloads
                ),
                "handling_characters": handling_characters,
                "movement_character_fraction": (simulation.moves / len(action)),
                "handling_character_fraction": (handling_characters / len(action)),
                "minimum_handling_characters": (minimum_handling_characters),
                "excess_characters_above_handling_lower_bound": (excess_characters),
                "instruction_lower_bound_efficiency": (minimum_handling_characters / len(action)),
                "normalized_cost_handling_lower_bound": normalized_cost(
                    minimum_handling_characters,
                    self._case.rows,
                    self._case.columns,
                ),
                "normalized_cost_gap_from_handling_lower_bound": (
                    excess_characters / (self._case.rows + self._case.columns - 1)
                ),
                "relocation_cycles": relocation_cycles,
                "relocation_characters": 4 * relocation_cycles,
                "relocation_cycles_per_shipment": (relocation_cycles / shipment_count),
                "terminal_reason": "complete_valid_solution",
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._case = None
        self._closed = True


__all__ = ["WarehousemanEnvironment"]
