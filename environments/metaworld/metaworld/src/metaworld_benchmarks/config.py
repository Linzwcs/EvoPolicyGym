"""Typed Host-selected MetaWorld MT task collections."""

from __future__ import annotations

from dataclasses import dataclass

METAWORLD_MT1_PROFILES = (
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
)

_MT10_TASKS = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "window-close-v3",
)
_MT1_SET = frozenset(METAWORLD_MT1_PROFILES)
_COLLECTION_PROFILES = frozenset({"mt10", "mt50", "custom"})


@dataclass(frozen=True, slots=True)
class MetaWorldConfig:
    """Configuration that fixes one MetaWorld task collection."""

    profile: str = "reach-v3"
    custom_tasks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.profile) is not str:
            raise TypeError("profile must be an exact string")
        if type(self.custom_tasks) is not tuple:
            raise TypeError("custom_tasks must be an exact tuple")
        if any(type(task) is not str for task in self.custom_tasks):
            raise TypeError("custom_tasks must contain exact strings")
        if self.profile not in _MT1_SET | _COLLECTION_PROFILES:
            raise ValueError(
                "profile must be a canonical MT1 task, 'mt10', 'mt50', "
                "or 'custom'"
            )
        if self.profile != "custom" and self.custom_tasks:
            raise ValueError(
                "custom_tasks may be set only for the custom profile"
            )
        if self.profile == "custom":
            if not self.custom_tasks:
                raise ValueError(
                    "custom_tasks must be non-empty for the custom profile"
                )
            if len(set(self.custom_tasks)) != len(self.custom_tasks):
                raise ValueError("custom_tasks must not contain duplicates")
            unknown = set(self.custom_tasks) - _MT1_SET
            if unknown:
                raise ValueError(
                    "custom_tasks contains unknown tasks: "
                    + ", ".join(sorted(unknown))
                )

    @property
    def task_names(self) -> tuple[str, ...]:
        if self.profile == "mt10":
            return _MT10_TASKS
        if self.profile == "mt50":
            return METAWORLD_MT1_PROFILES
        if self.profile == "custom":
            return self.custom_tasks
        return (self.profile,)

    @property
    def collection_name(self) -> str:
        return "mt1" if self.profile in _MT1_SET else self.profile


__all__ = ["METAWORLD_MT1_PROFILES", "MetaWorldConfig"]

