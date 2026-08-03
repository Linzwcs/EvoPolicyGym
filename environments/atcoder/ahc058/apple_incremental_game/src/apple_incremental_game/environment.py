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
            raise ValueError("Apple Incremental Game does not use an Episode scenario")
        self._seed = episode.environment_seed
        self._simulation: AppleSimulation | None = None
        self._started = False
        self._done = False
        self._closed = False
        self._wait_turn_count = 0
        self._total_spent = 0
        self._total_produced = 0
        self._peak_turn_production = 0
        self._last_upgrade_turn = -1
        self._upgrade_counts_by_level = [0] * LEVELS
        self._upgrade_counts_by_machine = [0] * MACHINE_IDS

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
        apples_before = self._simulation.apples
        upgrade_level = -1
        upgrade_machine_id = -1
        upgrade_cost = 0
        if upgrade is not None:
            upgrade_level, upgrade_machine_id = upgrade
            upgrade_cost = self._simulation.case.costs[upgrade_level][upgrade_machine_id] * (
                self._simulation.powers[upgrade_level][upgrade_machine_id] + 1
            )
        try:
            self._simulation.step(upgrade)
        except InvalidUpgrade:
            raise InvalidAction() from None

        self._done = self._simulation.done
        upgrade_made = upgrade is not None
        wait_action = not upgrade_made
        self._wait_turn_count += int(wait_action)
        if upgrade_made:
            self._last_upgrade_turn = self._simulation.turn
            self._upgrade_counts_by_level[upgrade_level] += 1
            self._upgrade_counts_by_machine[upgrade_machine_id] += 1
        self._total_spent += upgrade_cost
        apples_after_purchase = apples_before - upgrade_cost
        production = self._simulation.apples - apples_after_purchase
        if production < 0:
            raise RuntimeError("Apple production semantics drifted")
        self._total_produced += production
        self._peak_turn_production = max(
            self._peak_turn_production,
            production,
        )
        production_rate = sum(
            self._simulation.case.capacities[machine_id]
            * self._simulation.counts[0][machine_id]
            * self._simulation.powers[0][machine_id]
            for machine_id in range(MACHINE_IDS)
        )
        affordable = _affordable_upgrades(self._simulation)
        cheapest = (
            min(affordable, key=lambda item: (item[2], item[0], item[1])) if affordable else None
        )
        score = final_score(self._simulation.apples) if self._done else 0
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
                "upgrade_made_this_turn": upgrade_made,
                "wait_action_this_turn": wait_action,
                "upgrade_level": upgrade_level,
                "upgrade_machine_id": upgrade_machine_id,
                "upgrade_cost": upgrade_cost,
                "apples_before_action": apples_before,
                "apples_after_purchase": apples_after_purchase,
                "production_this_turn": production,
                "apple_net_change_this_turn": (self._simulation.apples - apples_before),
                "apples": self._simulation.apples,
                "score_if_ended_now": final_score(self._simulation.apples),
                "level_zero_production_rate": production_rate,
                "total_spent": self._total_spent,
                "total_produced": self._total_produced,
                "peak_turn_production": self._peak_turn_production,
                "total_upgrades": self._simulation.upgrades,
                "wait_turn_count": self._wait_turn_count,
                "wait_turn_fraction": (self._wait_turn_count / self._simulation.turn),
                "last_upgrade_turn": self._last_upgrade_turn,
                "turns_since_last_upgrade": (
                    self._simulation.turn - self._last_upgrade_turn
                    if self._last_upgrade_turn >= 0
                    else self._simulation.turn
                ),
                "upgrade_counts_by_level": list(self._upgrade_counts_by_level),
                "upgrade_counts_by_machine": list(self._upgrade_counts_by_machine),
                "machine_counts_by_level": [sum(row) for row in self._simulation.counts],
                "power_totals_by_level": [sum(row) for row in self._simulation.powers],
                "powered_machine_counts_by_level": [
                    sum(value > 0 for value in row) for row in self._simulation.powers
                ],
                "affordable_upgrade_count": len(affordable),
                "cheapest_affordable_upgrade_level": (cheapest[0] if cheapest is not None else -1),
                "cheapest_affordable_upgrade_machine_id": (
                    cheapest[1] if cheapest is not None else -1
                ),
                "cheapest_affordable_upgrade_cost": (cheapest[2] if cheapest is not None else None),
                "final_score": score if self._done else None,
                "terminal_reason": ("horizon_scored" if self._done else "in_progress"),
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


def _affordable_upgrades(
    simulation: AppleSimulation,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (
            level,
            machine_id,
            simulation.case.costs[level][machine_id] * (simulation.powers[level][machine_id] + 1),
        )
        for level in range(LEVELS)
        for machine_id in range(MACHINE_IDS)
        if simulation.case.costs[level][machine_id] * (simulation.powers[level][machine_id] + 1)
        <= simulation.apples
    )


__all__ = ["AppleIncrementalGameEnvironment"]
