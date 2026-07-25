"""Balatro-specific GameSpec implementation.

Provides ``balatro_game_spec()`` which returns a :class:`GameSpec` describing
Balatro's entity types (hand cards, jokers, consumables, shop items, pack cards),
21 action types, and feature dimensions.
"""

from __future__ import annotations

from jackdaw.env.game_spec import ActionTypeSpec, EntityTypeSpec, GameSpec
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
)

# Entity type indices (must match the order in entity_types tuple below)
HAND_CARD = 0
JOKER = 1
CONSUMABLE = 2
SHOP_ITEM = 3
PACK_CARD = 4


def balatro_game_spec() -> GameSpec:
    """Create the GameSpec for Balatro.

    The entity types, action types, and feature dimensions match the
    hardcoded values currently used throughout the policy and training code.
    """
    entity_types = (
        EntityTypeSpec(
            name="hand_card",
            feature_dim=D_PLAYING_CARD,
            max_count=8,
            has_catalog_id=False,
        ),
        EntityTypeSpec(
            name="joker",
            feature_dim=D_JOKER,
            max_count=5,
            has_catalog_id=True,
            catalog_size=NUM_CENTER_KEYS,
        ),
        EntityTypeSpec(
            name="consumable",
            feature_dim=D_CONSUMABLE,
            max_count=2,
            has_catalog_id=True,
            catalog_size=NUM_CENTER_KEYS,
        ),
        EntityTypeSpec(
            name="shop_item",
            feature_dim=D_SHOP,
            max_count=10,
            has_catalog_id=True,
            catalog_size=NUM_CENTER_KEYS,
        ),
        EntityTypeSpec(
            name="pack_card",
            feature_dim=D_PLAYING_CARD,
            max_count=5,
            has_catalog_id=False,
        ),
    )

    action_types = (
        # 0: PlayHand — select 1-5 cards to play
        ActionTypeSpec("play_hand", needs_entity_target=False, needs_card_select=True),
        # 1: Discard — select 1-5 cards to discard
        ActionTypeSpec("discard", needs_entity_target=False, needs_card_select=True),
        # 2: SelectBlind
        ActionTypeSpec("select_blind", needs_entity_target=False, needs_card_select=False),
        # 3: SkipBlind
        ActionTypeSpec("skip_blind", needs_entity_target=False, needs_card_select=False),
        # 4: CashOut
        ActionTypeSpec("cash_out", needs_entity_target=False, needs_card_select=False),
        # 5: Reroll
        ActionTypeSpec("reroll", needs_entity_target=False, needs_card_select=False),
        # 6: NextRound
        ActionTypeSpec("next_round", needs_entity_target=False, needs_card_select=False),
        # 7: SkipPack
        ActionTypeSpec("skip_pack", needs_entity_target=False, needs_card_select=False),
        # 8: BuyCard — target a shop item
        ActionTypeSpec(
            "buy_card",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=SHOP_ITEM,
        ),
        # 9: SellJoker — target a joker
        ActionTypeSpec(
            "sell_joker", needs_entity_target=True, needs_card_select=False, entity_type_index=JOKER
        ),
        # 10: SellConsumable — target a consumable
        ActionTypeSpec(
            "sell_consumable",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=CONSUMABLE,
        ),
        # 11: UseConsumable — target a consumable + select cards
        ActionTypeSpec(
            "use_consumable",
            needs_entity_target=True,
            needs_card_select=True,
            entity_type_index=CONSUMABLE,
        ),
        # 12: RedeemVoucher — target a shop item (voucher sub-type)
        ActionTypeSpec(
            "redeem_voucher",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=SHOP_ITEM,
        ),
        # 13: OpenBooster — target a shop item (booster sub-type)
        ActionTypeSpec(
            "open_booster",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=SHOP_ITEM,
        ),
        # 14: PickPackCard — target a pack card
        ActionTypeSpec(
            "pick_pack_card",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=PACK_CARD,
        ),
        # 15: SwapJokersLeft — target a joker
        ActionTypeSpec(
            "swap_jokers_left",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=JOKER,
        ),
        # 16: SwapJokersRight — target a joker
        ActionTypeSpec(
            "swap_jokers_right",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=JOKER,
        ),
        # 17: SwapHandLeft — target a hand card
        ActionTypeSpec(
            "swap_hand_left",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=HAND_CARD,
        ),
        # 18: SwapHandRight — target a hand card
        ActionTypeSpec(
            "swap_hand_right",
            needs_entity_target=True,
            needs_card_select=False,
            entity_type_index=HAND_CARD,
        ),
        # 19: SortHandRank
        ActionTypeSpec("sort_hand_rank", needs_entity_target=False, needs_card_select=False),
        # 20: SortHandSuit
        ActionTypeSpec("sort_hand_suit", needs_entity_target=False, needs_card_select=False),
    )

    spec = GameSpec(
        name="balatro",
        entity_types=entity_types,
        action_types=action_types,
        global_feature_dim=D_GLOBAL,
        max_card_select=5,
    )
    spec.validate()
    return spec


# Pre-computed convenience constants matching the old hardcoded values.
# These are derivable from the spec but kept as named exports for backward
# compatibility with code that imported them from ``action_heads``.
_SPEC = balatro_game_spec()
NEEDS_ENTITY: frozenset[int] = _SPEC.needs_entity_set
NEEDS_CARDS: frozenset[int] = _SPEC.needs_cards_set
NUM_ENTITY_TYPES: int = _SPEC.num_entity_types
