"""Executable modular Crafter scaffold; see PLAYER_GUIDE.md."""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from evopolicygym.policy import PolicyContext, PolicyValue, TensorValue

_MOVEMENT = (1, 2, 3, 4)
_OPPOSITE = {1: 2, 2: 1, 3: 4, 4: 3}
_ACTIONS = frozenset(range(17))

Position = tuple[int, int]


@dataclass(frozen=True)
class VisibleTile:
    """One optional semantic interpretation of a visible world tile."""

    offset: Position
    kind: str


@dataclass(frozen=True)
class VisibleEntity:
    """One optional semantic interpretation of a visible entity."""

    offset: Position
    kind: str


@dataclass(frozen=True)
class Perception:
    """Frame-derived evidence; unknown values remain explicit."""

    rgb: bytes
    health: int | None = None
    food: int | None = None
    drink: int | None = None
    energy: int | None = None
    daylight: float | None = None
    facing: Position | None = None
    inventory: tuple[tuple[str, int], ...] = ()
    tiles: tuple[VisibleTile, ...] = ()
    entities: tuple[VisibleEntity, ...] = ()


@dataclass
class WorldMemory:
    """Episode-local beliefs whose update rules are intentionally unfinished."""

    step: int = 0
    estimated_position: Position | None = None
    explored: set[Position] = field(default_factory=set)
    water_sources: set[Position] = field(default_factory=set)
    facility_locations: dict[str, set[Position]] = field(default_factory=dict)
    last_action: int | None = None
    last_proposal_source: str | None = None


@dataclass(frozen=True)
class ActionProposal:
    """A capability's observation-backed candidate for the next Action."""

    action: int
    source: str
    evidence: tuple[str, ...] = ()


class VisualTranslationModule:
    """Extension point for translating RGB pixels into semantic evidence."""

    def translate(self, frame: bytes) -> Perception:
        return Perception(rgb=frame)


class WorldMemoryModule:
    """Extension point for updating map, landmark, and facility beliefs."""

    def __init__(self) -> None:
        self.state = WorldMemory()

    def update(
        self,
        perception: Perception,
        *,
        last_action: int | None,
        last_proposal_source: str | None,
    ) -> WorldMemory:
        del perception
        self.state.step += 1
        self.state.last_action = last_action
        self.state.last_proposal_source = last_proposal_source
        return self.state


class CapabilityModule(Protocol):
    def propose(
        self,
        perception: Perception,
        memory: WorldMemory,
    ) -> ActionProposal | None: ...


class ExplorationModule:
    """Extension point for map coverage and landmark-aware navigation."""

    def propose(
        self,
        perception: Perception,
        memory: WorldMemory,
    ) -> ActionProposal | None:
        del perception, memory
        return None


class SurvivalModule:
    """Extension point for health, food, drink, energy, and rest decisions."""

    def propose(
        self,
        perception: Perception,
        memory: WorldMemory,
    ) -> ActionProposal | None:
        del perception, memory
        return None


class ProductionModule:
    """Extension point for gathering, facilities, and tool progression."""

    def propose(
        self,
        perception: Perception,
        memory: WorldMemory,
    ) -> ActionProposal | None:
        del perception, memory
        return None


class CombatDefenseModule:
    """Extension point for creature interactions and defensive actions."""

    def propose(
        self,
        perception: Perception,
        memory: WorldMemory,
    ) -> ActionProposal | None:
        del perception, memory
        return None


class Coordinator:
    """Extension point for resolving simultaneous capability proposals."""

    def select(
        self,
        proposals: tuple[ActionProposal, ...],
    ) -> ActionProposal | None:
        # A single proposal is unambiguous. The scaffold assigns no gameplay
        # priority when capabilities disagree.
        return proposals[0] if len(proposals) == 1 else None


class _SeededFallback:
    """Minimal legal behavior used only when capabilities make no decision."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)
        self._actions: deque[int] = deque()
        self._previous_direction: int | None = None

    def next_action(self) -> int:
        if not self._actions:
            candidates = tuple(
                direction
                for direction in _MOVEMENT
                if self._previous_direction is None
                or direction != _OPPOSITE[self._previous_direction]
            )
            direction = self._random.choice(candidates)
            self._actions.extend([direction] * self._random.randint(2, 5))
            self._actions.append(5)
            self._previous_direction = direction
        return self._actions.popleft()


class BaselinePolicy:
    def __init__(self, seed: int) -> None:
        self.visual = VisualTranslationModule()
        self.world = WorldMemoryModule()
        self.exploration = ExplorationModule()
        self.survival = SurvivalModule()
        self.production = ProductionModule()
        self.combat_defense = CombatDefenseModule()
        self.capabilities: tuple[CapabilityModule, ...] = (
            self.exploration,
            self.survival,
            self.production,
            self.combat_defense,
        )
        self.coordinator = Coordinator()
        self._fallback = _SeededFallback(seed)
        self._last_action: int | None = None
        self._last_proposal_source: str | None = None

    def act(self, observation: PolicyValue) -> int:
        if (
            type(observation) is not TensorValue
            or observation.dtype != "uint8"
            or observation.shape != (64, 64, 3)
        ):
            raise ValueError("Crafter observation is invalid")

        perception = self.visual.translate(observation.data)
        memory = self.world.update(
            perception,
            last_action=self._last_action,
            last_proposal_source=self._last_proposal_source,
        )
        proposals: list[ActionProposal] = []
        for capability in self.capabilities:
            proposal = capability.propose(perception, memory)
            if proposal is not None:
                proposals.append(proposal)

        selected = self.coordinator.select(tuple(proposals))
        if selected is None:
            action = self._fallback.next_action()
            source = "fallback"
        else:
            action = selected.action
            source = selected.source
        if type(action) is not int or action not in _ACTIONS:
            raise ValueError("Crafter proposal contains an invalid Action")

        self._last_action = action
        self._last_proposal_source = source
        return action


def make_policy(context: PolicyContext) -> BaselinePolicy:
    if context.environment_parameters.get("area") != [64, 64]:
        raise ValueError("Crafter area is invalid")
    if context.environment_parameters.get("image_size") != [64, 64]:
        raise ValueError("Crafter image size is invalid")
    return BaselinePolicy(context.policy_seed)
