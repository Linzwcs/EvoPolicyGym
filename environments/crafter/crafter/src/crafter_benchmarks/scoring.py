"""Shared scoring contract for Crafter survival-development v3."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .constants import ACHIEVEMENTS

SURVIVAL_CREDIT_PER_ALIVE_STEP = 1.0
VITAL_CREDIT_SCALE = 0.1
VITAL_NAMES = ("health", "food", "drink")
PROGRESS_CREDIT_MAX = 1_829.0
PRODUCTIVITY_CREDIT_MAX = 25.0

FIRST_UNLOCK_REWARDS = {
    "collect_coal": 16.0,
    "collect_diamond": 1_024.0,
    "collect_drink": 1.0,
    "collect_iron": 64.0,
    "collect_sapling": 1.0,
    "collect_stone": 16.0,
    "collect_wood": 1.0,
    "defeat_skeleton": 32.0,
    "defeat_zombie": 16.0,
    "eat_cow": 1.0,
    "eat_plant": 8.0,
    "make_iron_pickaxe": 256.0,
    "make_iron_sword": 256.0,
    "make_stone_pickaxe": 32.0,
    "make_stone_sword": 32.0,
    "make_wood_pickaxe": 8.0,
    "make_wood_sword": 8.0,
    "place_furnace": 32.0,
    "place_plant": 4.0,
    "place_stone": 16.0,
    "place_table": 4.0,
    "wake_up": 1.0,
}

REPEAT_EVENT_WEIGHTS = {
    "collect_drink": 3.0,
    "eat_cow": 3.0,
    "eat_plant": 3.0,
    "collect_wood": 1.0,
    "collect_sapling": 1.0,
    "collect_stone": 2.0,
    "collect_coal": 3.0,
    "collect_iron": 4.0,
    "collect_diamond": 6.0,
    "defeat_zombie": 3.0,
    "defeat_skeleton": 4.0,
    "place_plant": 2.0,
    "place_stone": 2.0,
    "place_table": 1.0,
    "place_furnace": 2.0,
}

REPEAT_EVENT_CAPS = {
    "collect_drink": 8,
    "eat_cow": 4,
    "eat_plant": 4,
    "collect_wood": 8,
    "collect_sapling": 4,
    "collect_stone": 8,
    "collect_coal": 4,
    "collect_iron": 3,
    "collect_diamond": 2,
    "defeat_zombie": 4,
    "defeat_skeleton": 2,
    "place_plant": 4,
    "place_stone": 8,
    "place_table": 2,
    "place_furnace": 2,
}

_REPEAT_WEIGHT_SUM = sum(REPEAT_EVENT_WEIGHTS.values())

if set(FIRST_UNLOCK_REWARDS) != set(ACHIEVEMENTS):
    raise RuntimeError("Crafter v3 first-unlock rewards must cover all achievements")
if sum(FIRST_UNLOCK_REWARDS.values()) != PROGRESS_CREDIT_MAX:
    raise RuntimeError("Crafter v3 first-unlock reward sum is invalid")
if set(REPEAT_EVENT_WEIGHTS) != set(REPEAT_EVENT_CAPS):
    raise RuntimeError("Crafter v3 repeat weights and caps do not align")
if _REPEAT_WEIGHT_SUM != 40.0:
    raise RuntimeError("Crafter v3 repeat weights must sum to 40")


def first_unlock_delta(unlocked: Sequence[str]) -> float:
    """Return the absolute reward for first unlocks on one transition."""

    if len(unlocked) != len(set(unlocked)):
        raise ValueError("Crafter v3 first unlocks contain duplicates")
    try:
        return sum(FIRST_UNLOCK_REWARDS[name] for name in unlocked)
    except KeyError as error:
        raise ValueError("Crafter v3 first unlock is invalid") from error


def vital_quality(vitals: Mapping[str, int]) -> float:
    """Return continuous weakest-vital quality without raw energy."""

    if set(vitals) != set(VITAL_NAMES):
        raise ValueError("Crafter v3 vitals are invalid")
    amounts: list[int] = []
    for name in VITAL_NAMES:
        amount = vitals[name]
        if type(amount) is not int or not 0 <= amount <= 9:
            raise ValueError("Crafter v3 vitals are invalid")
        amounts.append(amount)
    return min(amounts) / 9.0


def repeat_event_credit(name: str, total_events: int) -> float:
    """Return one event's bounded cumulative repeat credit."""

    if name not in REPEAT_EVENT_WEIGHTS:
        return 0.0
    if type(total_events) is not int or total_events < 0:
        raise ValueError("Crafter v3 event total is invalid")
    repeats = max(total_events - 1, 0)
    cap = REPEAT_EVENT_CAPS[name]
    normalized = math.log1p(min(repeats, cap)) / math.log1p(cap)
    return (
        PRODUCTIVITY_CREDIT_MAX
        * REPEAT_EVENT_WEIGHTS[name]
        / _REPEAT_WEIGHT_SUM
        * normalized
    )


def productivity_potential(event_totals: Mapping[str, int]) -> float:
    """Return bounded cumulative repeated-productivity potential."""

    if set(event_totals) != set(ACHIEVEMENTS):
        raise ValueError("Crafter v3 event totals are invalid")
    return sum(
        repeat_event_credit(name, event_totals[name])
        for name in REPEAT_EVENT_WEIGHTS
    )


def transition_score_components(
    *,
    terminated: bool,
    unlocked: Sequence[str],
    event_counts: Mapping[str, int],
    event_totals: Mapping[str, int],
    vitals: Mapping[str, int],
) -> tuple[dict[str, float], dict[str, int]]:
    """Score one transition and return updated cumulative event totals."""

    if type(terminated) is not bool:
        raise TypeError("terminated must be an exact bool")
    if set(event_totals) != set(ACHIEVEMENTS):
        raise ValueError("Crafter v3 event totals are invalid")
    updated = dict(event_totals)
    before = productivity_potential(updated)
    for name, count in event_counts.items():
        if (
            type(name) is not str
            or name not in ACHIEVEMENTS
            or type(count) is not int
            or count <= 0
        ):
            raise ValueError("Crafter v3 transition event counts are invalid")
        updated[name] += count
    after = productivity_potential(updated)
    alive = 0.0 if terminated else SURVIVAL_CREDIT_PER_ALIVE_STEP
    components = {
        "survival": alive,
        "vital": VITAL_CREDIT_SCALE * alive * vital_quality(vitals),
        "progress": first_unlock_delta(unlocked),
        "productivity": after - before,
    }
    if any(value < 0.0 or not math.isfinite(value) for value in components.values()):
        raise RuntimeError("Crafter v3 produced invalid score components")
    return components, updated


def score_delta(components: Mapping[str, float]) -> float:
    """Return the exact additive transition score."""

    if set(components) != {"survival", "vital", "progress", "productivity"}:
        raise ValueError("Crafter v3 score components are invalid")
    values = tuple(components.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("Crafter v3 score components are invalid")
    return math.fsum(float(value) for value in values)


__all__ = [
    "FIRST_UNLOCK_REWARDS",
    "PRODUCTIVITY_CREDIT_MAX",
    "PROGRESS_CREDIT_MAX",
    "REPEAT_EVENT_CAPS",
    "REPEAT_EVENT_WEIGHTS",
    "SURVIVAL_CREDIT_PER_ALIVE_STEP",
    "VITAL_CREDIT_SCALE",
    "VITAL_NAMES",
    "first_unlock_delta",
    "productivity_potential",
    "repeat_event_credit",
    "score_delta",
    "transition_score_components",
    "vital_quality",
]
