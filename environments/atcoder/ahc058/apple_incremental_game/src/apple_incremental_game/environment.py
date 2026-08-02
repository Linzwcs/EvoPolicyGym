"""One fresh independent Apple Incremental Game Environment per Episode."""

from __future__ import annotations

from evopolicygym.authoring import EpisodeSpec, InvalidAction, Step
from evopolicygym.policy import PolicyValue

from .simulation import (
    LEVELS,
    MACHINE_IDS,
    TURNS,
    AppleSimulation,
    InvalidUpgrade,
    final_score,
    generate_case,
)


class AppleIncrementalGameEnvironment:
    """Strict turn adapter for the public AHC058 dynamics."""

    def __init__(self, episode: EpisodeSpec) -> None:
        if type(episode) is not EpisodeSpec:
            raise TypeError("episode must be EpisodeSpec")
        if episode.scenario is not None:
            raise ValueError(
                "Apple Incremental Game does not use an Episode scenario"
            )
        self._seed = episode.environment_seed
        self._simulation: AppleSimulation | None = None
        self._started = False
        self._done = False
        self._closed = False

    def reset(self) -> PolicyValue:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._started:
            raise RuntimeError("Environment can be reset only once")
        self._simulation = AppleSimulation(generate_case(self._seed))
        self._started = True
        return _observation(self._simulation, include_initial=True)

    def step(self, action: PolicyValue) -> Step:
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._started or self._simulation is None:
            raise RuntimeError("Environment must be reset before step")
        if self._done:
            raise RuntimeError("Episode is already complete")

        upgrade = _decode_action(action)
        try:
            self._simulation.step(upgrade)
        except InvalidUpgrade:
            raise InvalidAction() from None

        self._done = self._simulation.done
        score = (
            final_score(self._simulation.apples)
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
                "apples": self._simulation.apples,
                "total_upgrades": self._simulation.upgrades,
                "final_score": score if self._done else None,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._simulation = None
        self._closed = True


def _decode_action(value: PolicyValue) -> tuple[int, int] | None:
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"upgrade"}:
        raise InvalidAction()
    raw_upgrade = value["upgrade"]
    if type(raw_upgrade) is not list or len(raw_upgrade) != 2:
        raise InvalidAction()
    level, machine_id = raw_upgrade
    if (
        type(level) is not int
        or type(machine_id) is not int
        or not 0 <= level < LEVELS
        or not 0 <= machine_id < MACHINE_IDS
    ):
        raise InvalidAction()
    return level, machine_id


def _observation(
    simulation: AppleSimulation,
    *,
    include_initial: bool,
) -> dict[str, PolicyValue]:
    initial: PolicyValue = None
    if include_initial:
        initial = {
            "capacities": list(simulation.case.capacities),
            "costs": [list(row) for row in simulation.case.costs],
        }
    return {
        "turn": simulation.turn,
        "turns_remaining": TURNS - simulation.turn,
        "apples": simulation.apples,
        "machines": [list(row) for row in simulation.counts],
        "powers": [list(row) for row in simulation.powers],
        "initial": initial,
    }


__all__ = ["AppleIncrementalGameEnvironment"]
