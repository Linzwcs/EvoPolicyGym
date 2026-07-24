"""Agent protocol and RandomAgent baseline for the Gymnasium environment."""

from __future__ import annotations

import random as _random
from typing import Protocol, runtime_checkable

import numpy as np

from jackdaw.env.action_space import (
    ActionMask,
    ActionType,
    FactoredAction,
)

# ---------------------------------------------------------------------------
# Agent Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Agent(Protocol):
    """Gymnasium-compatible agent protocol.

    Receives encoded observations and action masks (the same interface
    a policy network would use), and returns a :class:`FactoredAction`.
    """

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        """Select an action given observation and mask."""
        ...

    def reset(self) -> None:
        """Called at episode start."""
        ...


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


class RandomAgent:
    """Uniform random agent — picks randomly from legal actions.

    Samples a legal action type, then fills in entity/card targets
    as needed.  Always produces valid :class:`FactoredAction` instances.
    """

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        # Pick a random legal action type
        legal_types = np.nonzero(action_mask.type_mask)[0]
        if len(legal_types) == 0:
            return FactoredAction(action_type=ActionType.SelectBlind)

        at = int(_random.choice(legal_types))

        entity_target: int | None = None
        card_target: tuple[int, ...] | None = None

        # Entity target
        if at in action_mask.entity_masks:
            mask = action_mask.entity_masks[at]
            legal_entities = np.nonzero(mask)[0]
            if len(legal_entities) > 0:
                entity_target = int(_random.choice(legal_entities))

        # Card target for PlayHand, Discard
        if at in (ActionType.PlayHand, ActionType.Discard):
            legal_cards = np.nonzero(action_mask.card_mask)[0]
            if len(legal_cards) > 0:
                n = min(len(legal_cards), action_mask.max_card_select)
                count = _random.randint(action_mask.min_card_select, n)
                selected = sorted(_random.sample(list(legal_cards), count))
                card_target = tuple(int(i) for i in selected)

        # Card target for UseConsumable (if it needs targets)
        if at == ActionType.UseConsumable and entity_target is not None:
            gs = info.get("raw_state", {})
            consumables = gs.get("consumables", [])
            if entity_target < len(consumables):
                from jackdaw.env.action_space import get_consumable_target_info

                card = consumables[entity_target]
                min_cards, max_cards, needs = get_consumable_target_info(card)
                if needs:
                    legal_cards = np.nonzero(action_mask.card_mask)[0]
                    if len(legal_cards) >= min_cards:
                        n = min(len(legal_cards), max_cards)
                        count = _random.randint(min_cards, n)
                        selected = sorted(_random.sample(list(legal_cards), count))
                        card_target = tuple(int(i) for i in selected)

        return FactoredAction(
            action_type=at,
            card_target=card_target,
            entity_target=entity_target,
        )


__all__ = ["Agent", "RandomAgent"]
