"""One fresh independent Treant's Forest Environment per Episode."""

from __future__ import annotations

from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .simulation import (
    ForestSimulation,
    InvalidPlacement,
    Position,
    generate_case,
)

MAX_EPISODE_STEPS = 2_048


class TreantsForestEnvironment:
    """Strict stepwise adapter for the public AHC054 interaction rules."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Treant's Forest does not use an Episode scenario")
        self._seed = episode.environment_seed
        self._simulation: ForestSimulation | None = None
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        self._simulation = ForestSimulation(generate_case(self._seed))
        self._started = True
        return _observation(self._simulation, include_initial=True)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started or self._simulation is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        placements = _decode_action(action)
        try:
            self._simulation.step(placements)
        except InvalidPlacement:
            raise InvalidAction() from None

        terminated = self._simulation.done
        truncated = (
            self._simulation.turn >= MAX_EPISODE_STEPS and not terminated
        )
        self._done = terminated or truncated
        return Step(
            observation=_observation(
                self._simulation,
                include_initial=False,
            ),
            reward=1.0,
            terminated=terminated,
            truncated=truncated,
            metrics={
                "turns": self._simulation.turn,
                "placed_treants": self._simulation.placed_count,
                "revealed_cells": self._simulation.revealed_count,
                "flower_reached": terminated,
                "turn_cap_reached": truncated,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._simulation = None
        self._closed = True


def _decode_action(value: PolicyValue) -> tuple[Position, ...]:
    if type(value) is not dict or set(value) != {"placements"}:
        raise InvalidAction()
    raw_placements = value["placements"]
    if type(raw_placements) is not list or len(raw_placements) > 1_600:
        raise InvalidAction()
    placements: list[Position] = []
    for raw_cell in raw_placements:
        if type(raw_cell) is not list or len(raw_cell) != 2:
            raise InvalidAction()
        row, column = raw_cell
        if type(row) is not int or type(column) is not int:
            raise InvalidAction()
        placements.append((row, column))
    return tuple(placements)


def _observation(
    simulation: ForestSimulation,
    *,
    include_initial: bool,
) -> dict[str, PolicyValue]:
    case = simulation.case
    initial: PolicyValue = None
    if include_initial:
        initial = {
            "size": case.size,
            "entrance": list(case.entrance),
            "flower": list(case.flower),
            "trees": [
                [row, column]
                for row, column in sorted(case.initial_trees)
            ],
        }
    return {
        "turn": simulation.turn,
        "adventurer": list(simulation.position),
        "newly_revealed": [
            [row, column]
            for row, column in simulation.newly_revealed
        ],
        "revealed_cells": simulation.revealed_count,
        "placed_treants": simulation.placed_count,
        "initial": initial,
    }


__all__ = ["MAX_EPISODE_STEPS", "TreantsForestEnvironment"]
