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
        self._previous_position: Position | None = None
        self._previous_revealed_count = 0
        self._previous_flower_path_length = 0
        self._visited_positions: set[Position] = set()
        self._flower_first_revealed_turn = -1
        self._no_placement_turn_count = 0
        self._total_submitted_placements = 0
        self._best_flower_path_length = 0
        self._worst_flower_path_length = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        self._simulation = ForestSimulation(generate_case(self._seed))
        self._previous_position = self._simulation.position
        self._previous_revealed_count = self._simulation.revealed_count
        self._previous_flower_path_length = self._simulation.flower_path_length
        self._best_flower_path_length = self._previous_flower_path_length
        self._worst_flower_path_length = self._previous_flower_path_length
        self._visited_positions.add(self._simulation.position)
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
        previous_position = self._previous_position
        if previous_position is None:
            raise RuntimeError("Treant's Forest position history is unavailable")
        previous_path_length = self._previous_flower_path_length
        try:
            self._simulation.step(placements)
        except InvalidPlacement:
            raise InvalidAction() from None

        terminated = self._simulation.done
        truncated = self._simulation.turn >= MAX_EPISODE_STEPS and not terminated
        self._done = terminated or truncated
        newly_revealed_count = self._simulation.revealed_count - self._previous_revealed_count
        path_length = self._simulation.flower_path_length
        path_length_change = path_length - previous_path_length
        self._best_flower_path_length = min(
            self._best_flower_path_length,
            path_length,
        )
        self._worst_flower_path_length = max(
            self._worst_flower_path_length,
            path_length,
        )
        adventurer_revisit = self._simulation.position in self._visited_positions
        self._visited_positions.add(self._simulation.position)
        if self._simulation.flower_revealed and self._flower_first_revealed_turn < 0:
            self._flower_first_revealed_turn = self._simulation.turn
        placement_count = len(placements)
        no_placement = placement_count == 0
        self._no_placement_turn_count += int(no_placement)
        self._total_submitted_placements += placement_count
        size = self._simulation.case.size
        flower = self._simulation.case.flower
        position = self._simulation.position
        flower_manhattan_distance = abs(position[0] - flower[0]) + abs(position[1] - flower[1])
        previous_flower_manhattan_distance = abs(previous_position[0] - flower[0]) + abs(
            previous_position[1] - flower[1]
        )
        terminal_reason = (
            "flower_reached" if terminated else "turn_cap" if truncated else "in_progress"
        )
        self._previous_position = position
        self._previous_revealed_count = self._simulation.revealed_count
        self._previous_flower_path_length = path_length
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
                "remaining_turns": max(
                    MAX_EPISODE_STEPS - self._simulation.turn,
                    0,
                ),
                "placement_count_this_turn": placement_count,
                "no_placement_this_turn": no_placement,
                "no_placement_turn_count": self._no_placement_turn_count,
                "submitted_placement_count": self._total_submitted_placements,
                "mean_submitted_placements_per_turn": (
                    self._total_submitted_placements / self._simulation.turn
                ),
                "placed_treants": self._simulation.placed_count,
                "newly_revealed_cell_count": newly_revealed_count,
                "revealed_cells": self._simulation.revealed_count,
                "revealed_cell_fraction": (self._simulation.revealed_count / (size * size)),
                "legal_candidate_cell_count": (self._simulation.legal_candidate_count),
                "flower_revealed": self._simulation.flower_revealed,
                "flower_first_revealed_turn": (self._flower_first_revealed_turn),
                "flower_manhattan_distance": flower_manhattan_distance,
                "flower_manhattan_improvement_this_turn": (
                    previous_flower_manhattan_distance - flower_manhattan_distance
                ),
                "flower_path_length": path_length,
                "flower_path_length_change_this_turn": path_length_change,
                "best_flower_path_length": self._best_flower_path_length,
                "worst_flower_path_length": self._worst_flower_path_length,
                "adventurer_revisit": adventurer_revisit,
                "unique_adventurer_position_count": len(self._visited_positions),
                "score_so_far": float(self._simulation.turn),
                "flower_reached": terminated,
                "turn_cap_reached": truncated,
                "terminal_reason": terminal_reason,
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
            "trees": [[row, column] for row, column in sorted(case.initial_trees)],
        }
    return {
        "turn": simulation.turn,
        "adventurer": list(simulation.position),
        "newly_revealed": [[row, column] for row, column in simulation.newly_revealed],
        "revealed_cells": simulation.revealed_count,
        "placed_treants": simulation.placed_count,
        "initial": initial,
    }


__all__ = ["MAX_EPISODE_STEPS", "TreantsForestEnvironment"]
