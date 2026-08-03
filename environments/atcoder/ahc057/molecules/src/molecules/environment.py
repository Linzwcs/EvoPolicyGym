"""One fresh independent Molecules Environment per Episode."""

from __future__ import annotations

from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .simulation import (
    POINTS,
    SPACE_SIZE,
    TARGET_COMPONENTS,
    TARGET_SIZE,
    TURNS,
    Bond,
    InvalidBond,
    MoleculesCase,
    MoleculesSimulation,
    final_score,
    generate_case,
)


class MoleculesEnvironment:
    """Strict turn adapter for the public AHC057 dynamics."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError("Molecules does not use an Episode scenario")
        self._seed = episode.environment_seed
        self._case: MoleculesCase | None = None
        self._simulation: MoleculesSimulation | None = None
        self._started = False
        self._done = False
        self._closed = False
        self._bond_action_count = 0
        self._empty_bond_action_count = 0
        self._last_bond_turn = -1
        self._target_partition_first_ready_turn = -1
        self._maximum_action_cost = 0
        self._maximum_bond_cost = 0

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        self._case = generate_case(self._seed)
        self._simulation = MoleculesSimulation(self._case)
        self._started = True
        return _observation(self._simulation, include_initial=True)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started or self._simulation is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        bonds = _decode_action(action)
        bond_costs = self._simulation.bond_costs(bonds)
        try:
            action_cost = self._simulation.step(bonds)
        except InvalidBond:
            raise InvalidAction() from None

        self._done = self._simulation.done
        if action_cost != sum(bond_costs):
            raise RuntimeError("Molecules bond cost semantics drifted")
        bond_count = len(bonds)
        bond_action = bond_count > 0
        self._bond_action_count += int(bond_action)
        self._empty_bond_action_count += int(not bond_action)
        if bond_action:
            self._last_bond_turn = self._simulation.turn
        self._maximum_action_cost = max(
            self._maximum_action_cost,
            action_cost,
        )
        if bond_costs:
            self._maximum_bond_cost = max(
                self._maximum_bond_cost,
                max(bond_costs),
            )
        component_sizes = self._simulation.component_sizes
        target_partition_ready = component_sizes == (TARGET_SIZE,) * TARGET_COMPONENTS
        if target_partition_ready and self._target_partition_first_ready_turn < 0:
            self._target_partition_first_ready_turn = self._simulation.turn
        score = final_score(self._simulation.total_cost) if self._done else 0
        if self._done != (self._simulation.turn == TURNS):
            raise RuntimeError("Molecules horizon semantics drifted")
        if self._done and not target_partition_ready:
            raise RuntimeError("Molecules terminal partition semantics drifted")
        total_bonds = self._simulation.total_bonds
        required_bonds_remaining = self._simulation.component_count - TARGET_COMPONENTS
        component_histogram = self._simulation.component_size_histogram
        return Step(
            observation=_observation(
                self._simulation,
                include_initial=False,
            ),
            reward=float(score),
            terminated=self._done,
            metrics={
                "turn": self._simulation.turn,
                "turns_remaining": TURNS - self._simulation.turn,
                "bond_count_this_turn": bond_count,
                "bond_action_this_turn": bond_action,
                "bond_action_count": self._bond_action_count,
                "empty_bond_action_count": self._empty_bond_action_count,
                "empty_bond_action_fraction": (
                    self._empty_bond_action_count / self._simulation.turn
                ),
                "last_bond_turn": self._last_bond_turn,
                "action_cost": action_cost,
                "action_cost_per_bond": (action_cost / bond_count if bond_count else None),
                "minimum_bond_cost_this_turn": (min(bond_costs) if bond_costs else None),
                "maximum_bond_cost_this_turn": (max(bond_costs) if bond_costs else None),
                "maximum_action_cost": self._maximum_action_cost,
                "maximum_bond_cost": self._maximum_bond_cost,
                "total_cost": self._simulation.total_cost,
                "mean_cost_per_bond": (
                    self._simulation.total_cost / total_bonds if total_bonds else None
                ),
                "score_upper_bound_if_no_further_cost": final_score(self._simulation.total_cost),
                "total_bonds": total_bonds,
                "required_bonds_remaining": required_bonds_remaining,
                "bond_completion_fraction": (total_bonds / (POINTS - TARGET_COMPONENTS)),
                "components": self._simulation.component_count,
                "component_size_histogram": list(component_histogram),
                "singleton_component_count": component_histogram[0],
                "target_size_component_count": component_histogram[-1],
                "smallest_component_size": min(component_sizes),
                "largest_component_size": max(component_sizes),
                "target_partition_ready": target_partition_ready,
                "target_partition_first_ready_turn": (self._target_partition_first_ready_turn),
                "target_partition": self._done and target_partition_ready,
                "final_score": score if self._done else None,
                "terminal_reason": ("target_partition_scored" if self._done else "in_progress"),
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._case = None
        self._simulation = None
        self._closed = True


def _decode_action(value: PolicyValue) -> tuple[Bond, ...]:
    if type(value) is not dict or set(value) != {"bonds"}:
        raise InvalidAction()
    raw_bonds = value["bonds"]
    if type(raw_bonds) is not list or len(raw_bonds) > POINTS - TARGET_COMPONENTS:
        raise InvalidAction()
    bonds: list[Bond] = []
    for raw_bond in raw_bonds:
        if type(raw_bond) is not list or len(raw_bond) != 2:
            raise InvalidAction()
        first, second = raw_bond
        if (
            type(first) is not int
            or type(second) is not int
            or not 0 <= first < POINTS
            or not 0 <= second < POINTS
            or first == second
        ):
            raise InvalidAction()
        bonds.append((first, second))
    return tuple(bonds)


def _observation(
    simulation: MoleculesSimulation,
    *,
    include_initial: bool,
) -> dict[str, PolicyValue]:
    initial: PolicyValue = None
    if include_initial:
        initial = {
            "space_size": SPACE_SIZE,
            "target_components": TARGET_COMPONENTS,
            "target_size": TARGET_SIZE,
            "turns": TURNS,
        }
    return {
        "turn": simulation.turn,
        "turns_remaining": TURNS - simulation.turn,
        "positions": [[x, y] for x, y in simulation.positions],
        "velocities": [[vx, vy] for vx, vy in simulation.velocities],
        "components": list(simulation.component_labels),
        "component_count": simulation.component_count,
        "total_cost": simulation.total_cost,
        "initial": initial,
    }


__all__ = ["MoleculesEnvironment"]
