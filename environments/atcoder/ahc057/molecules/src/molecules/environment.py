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
        try:
            action_cost = self._simulation.step(bonds)
        except InvalidBond:
            raise InvalidAction() from None

        self._done = self._simulation.done
        score = (
            final_score(self._simulation.total_cost)
            if self._done
            else 0
        )
        return Step(
            observation=_observation(
                self._simulation,
                include_initial=False,
            ),
            reward=float(score),
            terminated=self._done,
            metrics={
                "turn": self._simulation.turn,
                "action_cost": action_cost,
                "total_cost": self._simulation.total_cost,
                "total_bonds": self._simulation.total_bonds,
                "components": self._simulation.component_count,
                "target_partition": (
                    self._done
                    and self._simulation.component_sizes
                    == (TARGET_SIZE,) * TARGET_COMPONENTS
                ),
                "final_score": score if self._done else None,
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
        "positions": [
            [x, y] for x, y in simulation.positions
        ],
        "velocities": [
            [vx, vy] for vx, vy in simulation.velocities
        ],
        "components": list(simulation.component_labels),
        "component_count": simulation.component_count,
        "total_cost": simulation.total_cost,
        "initial": initial,
    }


__all__ = ["MoleculesEnvironment"]
