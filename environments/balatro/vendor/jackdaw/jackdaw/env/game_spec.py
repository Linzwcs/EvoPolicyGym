"""Game-agnostic specification and runtime protocols for the RL pipeline.

Defines the ``GameSpec`` interface that any card game can implement to use
the generic policy network, encoder, and training loop.  Also provides
``GameObservation``, ``GameActionMask``, and ``GameEnvironment`` protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# FactoredAction — game-agnostic action representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactoredAction:
    """A single action in the factored representation.

    This is game-agnostic: just an action type index, optional entity
    target index, and optional card selection indices.

    Parameters
    ----------
    action_type:
        Index of the action type (meaning defined by the GameSpec).
    card_target:
        Tuple of hand card indices for card-selecting actions.
        ``None`` when not applicable.
    entity_target:
        Index into the relevant entity list.
        ``None`` for actions without entity targets.
    """

    action_type: int
    card_target: tuple[int, ...] | None = None
    entity_target: int | None = None


# ---------------------------------------------------------------------------
# Entity and action type specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityTypeSpec:
    """Describes one type of entity in the game."""

    name: str  # e.g. "hand_card", "joker", "minion"
    feature_dim: int  # dimension of raw feature vector
    max_count: int  # max entities of this type (for pre-allocation)
    has_catalog_id: bool  # whether entities have a categorical ID (like center_key)
    catalog_size: int = 0  # number of unique IDs if has_catalog_id


@dataclass(frozen=True)
class ActionTypeSpec:
    """Describes one type of action in the game."""

    name: str  # e.g. "play_hand", "summon_creature"
    needs_entity_target: bool  # requires selecting an entity
    needs_card_select: bool  # requires multi-selecting cards
    entity_type_index: int = -1  # which entity type the pointer targets (-1 = none)


@dataclass(frozen=True)
class GameSpec:
    """Complete specification of a game's RL interface.

    This is the central abstraction that decouples the neural network,
    training loop, and policy architecture from any specific game.
    """

    name: str
    entity_types: tuple[EntityTypeSpec, ...]
    action_types: tuple[ActionTypeSpec, ...]
    global_feature_dim: int  # dimension of global context vector
    max_card_select: int = 5  # maximum cards selectable in one action

    @property
    def num_entity_types(self) -> int:
        return len(self.entity_types)

    @property
    def num_action_types(self) -> int:
        return len(self.action_types)

    @property
    def needs_entity_set(self) -> frozenset[int]:
        """Action type indices that require an entity target."""
        return frozenset(i for i, a in enumerate(self.action_types) if a.needs_entity_target)

    @property
    def needs_cards_set(self) -> frozenset[int]:
        """Action type indices that require card selection."""
        return frozenset(i for i, a in enumerate(self.action_types) if a.needs_card_select)

    def entity_type_for_action(self, action_type_index: int) -> int:
        """Return the entity type index targeted by a given action type.

        Returns -1 if the action does not target an entity.
        """
        return self.action_types[action_type_index].entity_type_index

    def validate(self) -> None:
        """Check internal consistency. Raises ``ValueError`` on problems."""
        for i, at in enumerate(self.action_types):
            if at.needs_entity_target and at.entity_type_index < 0:
                raise ValueError(
                    f"Action type {i} ({at.name!r}) needs_entity_target=True "
                    f"but entity_type_index={at.entity_type_index}"
                )
            if at.entity_type_index >= len(self.entity_types):
                raise ValueError(
                    f"Action type {i} ({at.name!r}) entity_type_index="
                    f"{at.entity_type_index} out of range "
                    f"(only {len(self.entity_types)} entity types)"
                )
        for i, et in enumerate(self.entity_types):
            if et.has_catalog_id and et.catalog_size <= 0:
                raise ValueError(
                    f"Entity type {i} ({et.name!r}) has_catalog_id=True "
                    f"but catalog_size={et.catalog_size}"
                )


# ---------------------------------------------------------------------------
# Game-agnostic observation and action mask containers
# ---------------------------------------------------------------------------


@dataclass
class GameObservation:
    """Game-agnostic observation container.

    Attributes
    ----------
    global_context:
        Fixed-size global feature vector, shape ``(D_global,)``.
    entities:
        Maps entity type name to a variable-length feature array,
        ``entity_name -> (N_i, D_i)``.  Empty entity types should
        have shape ``(0, D_i)``.
    """

    global_context: np.ndarray  # (D_global,)
    entities: dict[str, np.ndarray]  # entity_name -> (N_i, D_i)


@dataclass
class GameActionMask:
    """Game-agnostic action mask container.

    Attributes
    ----------
    type_mask:
        Shape ``(num_action_types,)`` bool — which action types are legal.
    card_mask:
        Shape ``(N_hand,)`` bool — which hand cards can be selected.
    entity_masks:
        ``{action_type_index: np.ndarray}`` — for entity-targeted action
        types, which specific entity indices are legal targets.
    min_card_select:
        Minimum cards required for card-selecting actions.
    max_card_select:
        Maximum cards allowed for card-selecting actions.
    """

    type_mask: np.ndarray  # (num_action_types,) bool
    card_mask: np.ndarray  # (N_hand,) bool
    entity_masks: dict[int, np.ndarray]  # action_type -> (N_entity,) bool
    min_card_select: int = 0
    max_card_select: int = 5


# ---------------------------------------------------------------------------
# GameEnvironment protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GameEnvironment(Protocol):
    """Runtime interface that the RL pipeline needs from any game.

    Wraps a game instance and produces observations/masks in the
    game-agnostic format.
    """

    @property
    def spec(self) -> GameSpec: ...

    def reset(
        self, **kwargs: object
    ) -> tuple[GameObservation, GameActionMask, dict[str, object]]: ...

    def step(
        self, action: FactoredAction
    ) -> tuple[GameObservation, float, bool, bool, GameActionMask, dict[str, object]]: ...
