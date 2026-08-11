"""Self-contained scoring contract for Crafter's default LHS profile."""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from .constants import ACHIEVEMENTS

LHS_REWARD_PROFILE: Final[Literal["lhs"]] = (
    "lhs"
)
LHS_COMPONENT_NAMES = (
    "alive_survival",
    "vital_survival",
    "first_unlock",
    "maintenance_repeat",
    "productivity_repeat",
)
LHS_SURVIVAL_COMPONENT_NAMES = (
    "alive_survival",
    "vital_survival",
)
LHS_SECONDARY_COMPONENT_NAMES = (
    "first_unlock",
    "maintenance_repeat",
    "productivity_repeat",
)

# At full visible vitals, one 300-step day contributes 12 survival points.
LHS_ALIVE_ALPHA = 0.01
LHS_VITAL_ALPHA = 0.03
LHS_HEALTHY_STEP_CREDIT = LHS_ALIVE_ALPHA + LHS_VITAL_ALPHA
LHS_HEALTHY_WINDOW_STEPS = 300
LHS_HEALTHY_WINDOW_CREDIT = (
    LHS_HEALTHY_STEP_CREDIT * LHS_HEALTHY_WINDOW_STEPS
)

# The logarithm compresses the technology-stage spacing of the former raw
# weights. These values are part of LHS itself, not a selectable legacy
# profile.
_FIRST_UNLOCK_RAW_WEIGHTS = {
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
if set(_FIRST_UNLOCK_RAW_WEIGHTS) != set(ACHIEVEMENTS):
    raise RuntimeError("Crafter LHS unlock weights must cover all achievements")

LHS_FIRST_UNLOCK_BASE_CREDIT = 0.10
LHS_FIRST_UNLOCK_CREDITS = {
    name: (
        LHS_FIRST_UNLOCK_BASE_CREDIT
        * math.log1p(raw_weight)
        / math.log(2.0)
    )
    for name, raw_weight in _FIRST_UNLOCK_RAW_WEIGHTS.items()
}
LHS_FIRST_UNLOCK_CREDIT_MAX = math.fsum(
    LHS_FIRST_UNLOCK_CREDITS.values()
)

# Within any rolling 300-step window, productive repeats receive at most 20%
# of the corresponding first-unlock credit. The quota spreads that allowance
# across a useful cadence instead of paying it all on the first repeat.
LHS_REPEAT_WINDOW_STEPS = 300
LHS_PRODUCTIVITY_REPEAT_FRACTION = 0.20
LHS_PRODUCTIVITY_REPEAT_QUOTAS = {
    "collect_wood": 6,
    "collect_sapling": 3,
    "collect_stone": 6,
    "collect_coal": 3,
    "collect_iron": 2,
    "collect_diamond": 1,
    "defeat_zombie": 2,
    "defeat_skeleton": 1,
    "place_plant": 3,
}

# Maintenance keeps the absolute credit used by the selected profile: 1.2
# points per full rolling window, split equally between drink and food.
LHS_MAINTENANCE_WINDOW_CREDIT = 1.2
LHS_MAINTENANCE_RESOURCE_SHARES = {"drink": 0.50, "food": 0.50}
LHS_MAINTENANCE_RESTORE_UNIT_CAPS = {"drink": 15, "food": 12}
LHS_MAINTENANCE_RESTORE = {
    "collect_drink": ("drink", 1),
    "eat_cow": ("food", 6),
    "eat_plant": ("food", 4),
}
LHS_MAINTENANCE_UNIT_CREDITS = {
    resource: (
        LHS_MAINTENANCE_WINDOW_CREDIT
        * LHS_MAINTENANCE_RESOURCE_SHARES[resource]
        / cap
    )
    for resource, cap in LHS_MAINTENANCE_RESTORE_UNIT_CAPS.items()
}

# Survival alone selects the weak tail. Secondary development and maintenance
# components use ordinary means and cannot move an achievement-rich short
# Episode out of that tail.
LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT = 0.75
LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT = 0.25
LHS_FEEDBACK_SURVIVAL_TAIL_FRACTION = 0.25
LHS_POLICY_FAILURE_RETURN = 0.0
LHS_SURVIVAL_THRESHOLDS = (150, 200, 250, 300, 400)
LHS_VITAL_AGE_BANDS = (
    ("0-99", 0, 100),
    ("100-199", 100, 200),
    ("200-299", 200, 300),
    ("300+", 300, None),
)


def _empty_event_totals() -> dict[str, int]:
    return {name: 0 for name in ACHIEVEMENTS}


def _empty_productivity_windows() -> dict[str, deque[int]]:
    return {
        name: deque() for name in LHS_PRODUCTIVITY_REPEAT_QUOTAS
    }


def _empty_maintenance_windows() -> dict[str, deque[tuple[int, int]]]:
    return {
        resource: deque()
        for resource in LHS_MAINTENANCE_RESTORE_UNIT_CAPS
    }


@dataclass(slots=True)
class LHSScoringState:
    """Replayable per-Episode state for LHS transition scoring."""

    step_index: int = 0
    event_totals: dict[str, int] = field(default_factory=_empty_event_totals)
    previous_vitals: dict[str, int] = field(
        default_factory=lambda: {"health": 9, "food": 9, "drink": 9}
    )
    productivity_windows: dict[str, deque[int]] = field(
        default_factory=_empty_productivity_windows
    )
    maintenance_windows: dict[str, deque[tuple[int, int]]] = field(
        default_factory=_empty_maintenance_windows
    )

    def transition(
        self,
        *,
        terminated: bool,
        unlocked: Sequence[str],
        event_counts: Mapping[str, int],
        vitals: Mapping[str, int],
    ) -> tuple[dict[str, float], dict[str, object]]:
        """Score one transition and mutate this state exactly once."""

        if type(terminated) is not bool:
            raise TypeError("terminated must be an exact bool")
        if len(unlocked) != len(set(unlocked)):
            raise ValueError("Crafter LHS first unlocks contain duplicates")
        if any(name not in ACHIEVEMENTS for name in unlocked):
            raise ValueError("Crafter LHS first unlock is invalid")
        if set(vitals) != {"health", "food", "drink"}:
            raise ValueError("Crafter LHS vitals are invalid")

        current_vitals: dict[str, int] = {}
        for name in ("health", "food", "drink"):
            amount = vitals[name]
            if type(amount) is not int or not 0 <= amount <= 9:
                raise ValueError("Crafter LHS vitals are invalid")
            current_vitals[name] = amount

        counts: dict[str, int] = {}
        for name, count in event_counts.items():
            if (
                type(name) is not str
                or name not in ACHIEVEMENTS
                or type(count) is not int
                or count <= 0
            ):
                raise ValueError("Crafter LHS transition events are invalid")
            counts[name] = count
        expected_unlocked = {
            name for name in counts if self.event_totals[name] == 0
        }
        if set(unlocked) != expected_unlocked:
            raise ValueError(
                "Crafter LHS first unlocks do not match event history"
            )

        self.step_index += 1
        cutoff = self.step_index - LHS_REPEAT_WINDOW_STEPS
        for productivity_window in self.productivity_windows.values():
            while productivity_window and productivity_window[0] <= cutoff:
                productivity_window.popleft()
        for maintenance_window in self.maintenance_windows.values():
            while maintenance_window and maintenance_window[0][0] <= cutoff:
                maintenance_window.popleft()

        first_unlock = math.fsum(
            LHS_FIRST_UNLOCK_CREDITS[name] for name in unlocked
        )
        productivity_repeat = 0.0
        credited_productivity: dict[str, int] = {}
        credited_maintenance = {"drink": 0, "food": 0}

        for name, count in counts.items():
            repeats = count - int(self.event_totals[name] == 0)
            if repeats <= 0:
                continue

            if name in LHS_PRODUCTIVITY_REPEAT_QUOTAS:
                window = self.productivity_windows[name]
                quota = LHS_PRODUCTIVITY_REPEAT_QUOTAS[name]
                credited = min(repeats, max(quota - len(window), 0))
                if credited:
                    window.extend([self.step_index] * credited)
                    productivity_repeat += (
                        credited
                        * LHS_PRODUCTIVITY_REPEAT_FRACTION
                        * LHS_FIRST_UNLOCK_CREDITS[name]
                        / quota
                    )
                    credited_productivity[name] = credited

            maintenance = LHS_MAINTENANCE_RESTORE.get(name)
            if maintenance is None:
                continue
            resource, nominal_restore = maintenance
            possible_restore = min(
                nominal_restore,
                9 - self.previous_vitals[resource],
            )
            if possible_restore <= 0:
                continue
            resource_window = self.maintenance_windows[resource]
            used_units = sum(units for _, units in resource_window)
            available_units = max(
                LHS_MAINTENANCE_RESTORE_UNIT_CAPS[resource] - used_units,
                0,
            )
            credited_units = min(possible_restore, available_units)
            if credited_units:
                resource_window.append((self.step_index, credited_units))
                credited_maintenance[resource] += credited_units

        for name, count in counts.items():
            self.event_totals[name] += count
        self.previous_vitals = current_vitals

        components = {
            "alive_survival": (
                0.0 if terminated else LHS_ALIVE_ALPHA
            ),
            "vital_survival": (
                0.0
                if terminated
                else LHS_VITAL_ALPHA * min(current_vitals.values()) / 9.0
            ),
            "first_unlock": first_unlock,
            "maintenance_repeat": math.fsum(
                credited_maintenance[resource]
                * LHS_MAINTENANCE_UNIT_CREDITS[resource]
                for resource in credited_maintenance
            ),
            "productivity_repeat": productivity_repeat,
        }
        if any(
            value < 0.0 or not math.isfinite(value)
            for value in components.values()
        ):
            raise RuntimeError("Crafter LHS produced invalid score components")
        return components, {
            "maintenance_credited_units": credited_maintenance,
            "productivity_credited_events": credited_productivity,
        }


def lhs_score_delta(components: Mapping[str, float]) -> float:
    """Return the exact additive LHS transition score."""

    if set(components) != set(LHS_COMPONENT_NAMES):
        raise ValueError("Crafter LHS score components are invalid")
    values = tuple(components.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ):
        raise ValueError("Crafter LHS score components are invalid")
    return math.fsum(float(value) for value in values)


def lhs_feedback_score(
    survival_returns: Sequence[float],
    secondary_returns: Sequence[float],
) -> tuple[float, float, float, int, float]:
    """Aggregate the survival lower tail separately from secondary progress."""

    survival = tuple(float(value) for value in survival_returns)
    secondary = tuple(float(value) for value in secondary_returns)
    if (
        not survival
        or len(survival) != len(secondary)
        or any(not math.isfinite(value) or value < 0.0 for value in survival)
        or any(not math.isfinite(value) or value < 0.0 for value in secondary)
    ):
        raise ValueError("Crafter LHS returns must be finite and aligned")
    tail_count = max(
        1,
        math.ceil(LHS_FEEDBACK_SURVIVAL_TAIL_FRACTION * len(survival)),
    )
    mean_survival = statistics.fmean(survival)
    lower_survival = statistics.fmean(sorted(survival)[:tail_count])
    mean_secondary = statistics.fmean(secondary)
    score = math.fsum(
        (
            LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT * mean_survival,
            LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT * lower_survival,
            mean_secondary,
        )
    )
    return mean_survival, lower_survival, mean_secondary, tail_count, score


__all__ = [
    "LHS_ALIVE_ALPHA",
    "LHS_COMPONENT_NAMES",
    "LHS_FEEDBACK_SURVIVAL_LOWER_TAIL_WEIGHT",
    "LHS_FEEDBACK_SURVIVAL_MEAN_WEIGHT",
    "LHS_FEEDBACK_SURVIVAL_TAIL_FRACTION",
    "LHS_FIRST_UNLOCK_BASE_CREDIT",
    "LHS_FIRST_UNLOCK_CREDITS",
    "LHS_FIRST_UNLOCK_CREDIT_MAX",
    "LHS_HEALTHY_STEP_CREDIT",
    "LHS_HEALTHY_WINDOW_CREDIT",
    "LHS_HEALTHY_WINDOW_STEPS",
    "LHS_MAINTENANCE_RESOURCE_SHARES",
    "LHS_MAINTENANCE_RESTORE",
    "LHS_MAINTENANCE_RESTORE_UNIT_CAPS",
    "LHS_MAINTENANCE_UNIT_CREDITS",
    "LHS_MAINTENANCE_WINDOW_CREDIT",
    "LHS_POLICY_FAILURE_RETURN",
    "LHS_PRODUCTIVITY_REPEAT_FRACTION",
    "LHS_PRODUCTIVITY_REPEAT_QUOTAS",
    "LHS_REPEAT_WINDOW_STEPS",
    "LHS_REWARD_PROFILE",
    "LHS_SECONDARY_COMPONENT_NAMES",
    "LHS_SURVIVAL_COMPONENT_NAMES",
    "LHS_SURVIVAL_THRESHOLDS",
    "LHS_VITAL_AGE_BANDS",
    "LHS_VITAL_ALPHA",
    "LHSScoringState",
    "lhs_feedback_score",
    "lhs_score_delta",
]
