"""Entity-based observation encoding for neural network input.

Converts the raw engine ``game_state`` dict into structured numeric arrays.
Uses variable-length entity lists (no artificial caps) following the AlphaStar
entity encoding pattern.  Each entity type gets a fixed-width feature vector;
the number of entities varies per timestep.

The only external dependency is numpy (standard for RL).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card import Card
from jackdaw.engine.consumables import can_use_consumable
from jackdaw.engine.data.hands import HAND_BASE, HandType
from jackdaw.engine.hand_eval import (
    get_best_hand,
    get_hand_eval_flags,
)
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.env.game_spec import GameObservation

# ---------------------------------------------------------------------------
# Center key → integer ID mapping (loaded once at module init)
# ---------------------------------------------------------------------------

_CENTERS_PATH = Path(__file__).resolve().parent.parent / "engine" / "data" / "centers.json"

_CENTER_KEY_TO_ID: dict[str, int] = {}  # 0 = unknown/padding


def _load_center_ids() -> dict[str, int]:
    """Build a contiguous int mapping from centers.json keys."""
    with open(_CENTERS_PATH) as f:
        data: dict[str, Any] = json.load(f)
    mapping: dict[str, int] = {}
    for i, key in enumerate(sorted(data.keys()), start=1):
        mapping[key] = i
    return mapping


_CENTER_KEY_TO_ID = _load_center_ids()

# Number of distinct center keys (for embedding table sizing)
NUM_CENTER_KEYS: int = len(_CENTER_KEY_TO_ID)  # ~299


def center_key_id(key: str) -> int:
    """Map a center_key to its integer ID (0 for unknown)."""
    return _CENTER_KEY_TO_ID.get(key, 0)


# ---------------------------------------------------------------------------
# Categorical index tables
# ---------------------------------------------------------------------------

_SUIT_IDX: dict[str, int] = {
    "Hearts": 0,
    "Diamonds": 1,
    "Clubs": 2,
    "Spades": 3,
}

_RANK_IDX: dict[str, int] = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "10": 8,
    "Jack": 9,
    "Queen": 10,
    "King": 11,
    "Ace": 12,
}

_RANK_CHIPS: dict[str, int] = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "Jack": 10,
    "Queen": 10,
    "King": 10,
    "Ace": 11,
}

_ENHANCEMENT_IDX: dict[str, int] = {
    "none": 0,
    "c_base": 0,
    "m_bonus": 1,
    "m_mult": 2,
    "m_wild": 3,
    "m_glass": 4,
    "m_steel": 5,
    "m_stone": 6,
    "m_gold": 7,
    "m_lucky": 8,
}

_EDITION_IDX_MAP: dict[str, int] = {
    "none": 0,
    "foil": 1,
    "holo": 2,
    "polychrome": 3,
    "negative": 4,
}

_SEAL_IDX: dict[str | None, int] = {
    None: 0,
    "none": 0,
    "Gold": 1,
    "Red": 2,
    "Blue": 3,
    "Purple": 4,
}

_PHASE_IDX: dict[str, int] = {
    GamePhase.BLIND_SELECT: 0,
    GamePhase.SELECTING_HAND: 1,
    GamePhase.ROUND_EVAL: 2,
    GamePhase.SHOP: 3,
    GamePhase.PACK_OPENING: 4,
    GamePhase.GAME_OVER: 5,
}

_BLIND_ON_DECK_IDX: dict[str | None, int] = {
    None: 0,
    "Small": 1,
    "Big": 2,
    "Boss": 3,
}

_CARD_SET_IDX: dict[str, int] = {
    "": 0,
    "Default": 0,
    "Enhanced": 1,
    "Joker": 2,
    "Tarot": 3,
    "Planet": 4,
    "Spectral": 5,
    "Voucher": 6,
    "Booster": 7,
    "Back": 8,
    "Edition": 9,
}

# Hand types in canonical order for the global context vector
_HAND_TYPES: list[HandType] = list(HandType)
NUM_HAND_TYPES: int = len(_HAND_TYPES)  # 12

# Voucher keys in canonical order for binary vector
_VOUCHER_KEYS: list[str] = [
    "v_antimatter",
    "v_blank",
    "v_clearance_sale",
    "v_crystal_ball",
    "v_directors_cut",
    "v_glow_up",
    "v_grabber",
    "v_hieroglyph",
    "v_hone",
    "v_illusion",
    "v_liquidation",
    "v_magic_trick",
    "v_money_tree",
    "v_nacho_tong",
    "v_observatory",
    "v_omen_globe",
    "v_overstock_norm",
    "v_overstock_plus",
    "v_paint_brush",
    "v_palette",
    "v_petroglyph",
    "v_planet_merchant",
    "v_planet_tycoon",
    "v_recyclomancy",
    "v_reroll_glut",
    "v_reroll_surplus",
    "v_retcon",
    "v_seed_money",
    "v_tarot_merchant",
    "v_tarot_tycoon",
    "v_telescope",
    "v_wasteful",
]
NUM_VOUCHERS: int = len(_VOUCHER_KEYS)  # 32
_VOUCHER_IDX: dict[str, int] = {k: i for i, k in enumerate(_VOUCHER_KEYS)}

# Tag keys in canonical order for binary vector
_TAG_KEYS: list[str] = [
    "tag_boss",
    "tag_buffoon",
    "tag_charm",
    "tag_coupon",
    "tag_d_six",
    "tag_double",
    "tag_economy",
    "tag_ethereal",
    "tag_foil",
    "tag_garbage",
    "tag_handy",
    "tag_holo",
    "tag_investment",
    "tag_juggle",
    "tag_meteor",
    "tag_negative",
    "tag_orbital",
    "tag_polychrome",
    "tag_rare",
    "tag_skip",
    "tag_standard",
    "tag_top_up",
    "tag_uncommon",
    "tag_voucher",
]
NUM_TAGS: int = len(_TAG_KEYS)  # 24
_TAG_IDX: dict[str, int] = {k: i for i, k in enumerate(_TAG_KEYS)}

# Suit/rank indices for discard pile histogram (4 suits × 13 ranks = 52)
NUM_SUITS: int = 4
NUM_RANKS: int = 13
D_DISCARD_HISTOGRAM: int = NUM_SUITS * NUM_RANKS  # 52

# Boss blind debuff categories (for one-hot encoding)
_DEBUFF_SUIT_IDX: dict[str, int] = {
    "Hearts": 0,
    "Diamonds": 1,
    "Clubs": 2,
    "Spades": 3,
}

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

D_PLAYING_CARD: int = 15
D_JOKER: int = 15
D_CONSUMABLE: int = 7
D_SHOP: int = 9
# Global layout:
#   [0:6]      phase one-hot (6)
#   [6:10]     blind_on_deck one-hot (4)
#   [10:30]    scalar features (20)
#   [30:90]    hand_levels 12×5 (60)
#   [90:122]   vouchers_owned binary (32)
#   [122:130]  blind_effect features (8)
#   [130:133]  round_position one-hot (3)
#   [133:135]  round_progress (2)
#   [135:159]  tags binary (24)
#   [159:211]  discard_pile histogram (52)
#   --- NEW: strategic features ---
#   [211:223]  hand_type_indicators (12) — binary, which hand types current hand forms
#   [223:226]  base_score (3) — log(chips), log(mult), log(chips×mult) for best hand
#   [226]      score_to_blind_ratio (1) — log(base_score)/log(blind_chips), clamped
#   [227]      flush_proximity (1) — max suit count in hand / 5 (continuous)
#   [228]      straight_proximity (1) — max consecutive run length / 5 (continuous)
#   [229]      interest_earned (1) — current interest bracket, normalized
#   [230]      flush_draw_outs (1) — P(drawing suit-completing card) from deck
#   [231]      straight_draw_outs (1) — P(drawing rank-completing card) from deck
#   [232]      debuff_hand_fraction (1) — fraction of hand cards debuffed by boss blind
#   [233]      spendable_above_interest (1) — money above interest floor, log-scaled
#   [234]      urgency (1) — blind_remaining / (base_score × hands_left), clamped
_D_BASE: int = 6 + 4 + 20 + NUM_HAND_TYPES * 5  # 90
_D_OLD_GLOBAL: int = _D_BASE + NUM_VOUCHERS + 8 + 3 + 2 + NUM_TAGS + D_DISCARD_HISTOGRAM  # 211
_D_STRATEGIC: int = NUM_HAND_TYPES + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # 24
D_GLOBAL: int = _D_OLD_GLOBAL + _D_STRATEGIC  # 235


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LOG2 = math.log(2.0)


def _log_scale(x: float) -> float:
    """Log-scale large values: sign(x) * log2(1 + |x|)."""
    if x >= 0:
        return math.log2(1.0 + x)
    return -math.log2(1.0 - x)


def _log_scale_arr(arr: np.ndarray) -> np.ndarray:
    """Vectorized log-scale: sign(x) * log2(1 + |x|)."""
    signs = np.sign(arr)
    return signs * np.log2(1.0 + np.abs(arr))


def _edition_idx(edition: dict[str, Any] | None) -> int:
    """Map edition dict to integer index."""
    if not edition:
        return 0
    if edition.get("foil"):
        return 1
    if edition.get("holo"):
        return 2
    if edition.get("polychrome"):
        return 3
    if edition.get("negative"):
        return 4
    return 0


def _one_hot(idx: int, n: int) -> list[float]:
    """Return a one-hot list of length n with 1.0 at idx."""
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Entity encoding functions
# ---------------------------------------------------------------------------


def encode_playing_card(
    card: Card,
    position: int,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a playing card as a fixed-size feature vector.

    Returns shape ``(D_PLAYING_CARD,)`` float32 array.

    Features (15):
        0: rank_id (ordinal 2-14, normalized /14)
        1: suit (ordinal 0-3, normalized /3)
        2: chip_value (normalized /11)
        3: enhancement (ordinal 0-8, normalized /8)
        4: edition (ordinal 0-4, normalized /4)
        5: seal (ordinal 0-4, normalized /4)
        6: debuffed (0/1)
        7: face_down (0/1)
        8: is_face_card (0/1)
        9: is_scoring (0/1 — all cards if splash active)
       10: bonus_chips (log-scaled)
       11: times_played (log-scaled)
       12: position_in_hand (normalized /20)
       13: is_best_hand_card (0/1 — contributes to best detected hand)
       14: reserved/padding (0)
    """
    v = np.zeros(D_PLAYING_CARD, dtype=np.float32)

    face_down = card.facing == "back"

    if face_down or card.base is None:
        # Card exists but attributes hidden
        v[7] = 1.0
        v[12] = position / 20.0
        return v

    rank_str = card.base.rank.value
    suit_str = card.base.suit.value

    v[0] = card.base.id / 14.0
    v[1] = _SUIT_IDX.get(suit_str, 0) / 3.0
    v[2] = _RANK_CHIPS.get(rank_str, 0) / 11.0
    v[3] = _ENHANCEMENT_IDX.get(card.center_key, 0) / 8.0
    v[4] = _edition_idx(card.edition) / 4.0
    v[5] = _SEAL_IDX.get(card.seal, 0) / 4.0
    v[6] = float(card.debuff)
    v[7] = 0.0  # not face down
    v[8] = float(card.base.id in (11, 12, 13))  # J, Q, K
    v[9] = float(bool(gs.get("splash", 0)) or not card.debuff)
    v[10] = _log_scale(card.ability.get("bonus", 0) + card.ability.get("perma_bonus", 0))
    v[11] = _log_scale(card.base.times_played)
    v[12] = position / 20.0

    return v


def encode_joker(
    card: Card,
    position: int,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a joker as a fixed-size feature vector.

    Returns shape ``(D_JOKER,)`` float32 array.

    Features (15):
        0: center_key_id (normalized / NUM_CENTER_KEYS)
        1: rarity (ordinal 1-4, normalized /4)
        2: edition (ordinal 0-4, normalized /4)
        3: sell_value (log-scaled)
        4: eternal (0/1)
        5: perishable (0/1)
        6: perish_tally (normalized /5)
        7: rental (0/1)
        8: debuffed (0/1)
        9: position (normalized /20)
       10: ability_mult (log-scaled)
       11: ability_x_mult (raw — typically 1-5)
       12: ability_chips (log-scaled)
       13: ability_extra (log-scaled — meaning varies per joker)
       14: condition_met (0/1 — simplified: not debuffed)
    """
    v = np.zeros(D_JOKER, dtype=np.float32)

    v[0] = center_key_id(card.center_key) / max(NUM_CENTER_KEYS, 1)
    card.ability.get("extra", {})
    # Rarity from center data — ability dict doesn't store it, use cost heuristic
    # or look it up. For now use a simple cost-based proxy.
    v[1] = min(card.base_cost / 20.0, 1.0) if card.base_cost else 0.0
    v[2] = _edition_idx(card.edition) / 4.0
    v[3] = _log_scale(card.sell_cost)
    v[4] = float(card.eternal)
    v[5] = float(card.perishable)
    v[6] = card.perish_tally / 5.0 if card.perishable else 1.0
    v[7] = float(card.rental)
    v[8] = float(card.debuff)
    v[9] = position / 20.0
    v[10] = _log_scale(card.ability.get("mult", 0) + card.ability.get("t_mult", 0))
    v[11] = card.ability.get("x_mult", 1.0)
    v[12] = _log_scale(card.ability.get("t_chips", 0) + card.ability.get("bonus", 0))

    # ability_extra — flatten to a single float
    extra = card.ability.get("extra")
    if isinstance(extra, (int, float)):
        v[13] = _log_scale(extra)
    elif isinstance(extra, dict):
        # Sum numeric values as a generic signal
        total = 0.0
        for val in extra.values():
            if isinstance(val, (int, float)):
                total += val
        v[13] = _log_scale(total)

    v[14] = float(not card.debuff)

    return v


def encode_consumable(
    card: Card,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a consumable as a fixed-size feature vector.

    Returns shape ``(D_CONSUMABLE,)`` float32 array.

    Features (7):
        0: center_key_id (normalized)
        1: card_set (ordinal 0-9, normalized /9)
        2: sell_value (log-scaled)
        3: can_use (0/1)
        4: needs_targets (0/1)
        5: max_targets (normalized /5)
        6: min_targets (normalized /5)
    """
    v = np.zeros(D_CONSUMABLE, dtype=np.float32)

    v[0] = center_key_id(card.center_key) / max(NUM_CENTER_KEYS, 1)
    card_set = card.ability.get("set", "")
    v[1] = _CARD_SET_IDX.get(card_set, 0) / 9.0
    v[2] = _log_scale(card.sell_cost)

    # can_use — check without highlighting (conservative: some need targets)

    hand_cards: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    consumables: list[Card] = gs.get("consumables", [])
    can_use = can_use_consumable(
        card,
        hand_cards=hand_cards,
        jokers=jokers,
        consumables=consumables,
        consumable_limit=gs.get("consumable_slots", 2),
        joker_limit=gs.get("joker_slots", 5),
    )
    v[3] = float(can_use)

    # Targeting info from consumable config
    cfg = card.ability.get("consumeable", {})
    if not isinstance(cfg, dict):
        cfg = {}
    max_h = cfg.get("max_highlighted", 0)
    min_h = cfg.get("min_highlighted", 0)
    v[4] = float(max_h > 0)
    v[5] = min(max_h, 5) / 5.0
    v[6] = min(min_h, 5) / 5.0

    return v


def encode_shop_item(
    card: Card,
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode a shop item as a fixed-size feature vector.

    Returns shape ``(D_SHOP,)`` float32 array.

    Features (9):
        0: center_key_id (normalized)
        1: card_set (ordinal, normalized /9)
        2: cost (log-scaled)
        3: affordable (0/1)
        4: has_slot (0/1)
        5: edition (ordinal 0-4, normalized /4)
        6: eternal (0/1)
        7: perishable (0/1)
        8: rental (0/1)
    """
    v = np.zeros(D_SHOP, dtype=np.float32)

    v[0] = center_key_id(card.center_key) / max(NUM_CENTER_KEYS, 1)
    card_set = card.ability.get("set", "")
    v[1] = _CARD_SET_IDX.get(card_set, 0) / 9.0
    v[2] = _log_scale(card.cost)

    dollars = gs.get("dollars", 0)
    v[3] = float(card.cost <= dollars)

    # Has slot check
    jokers: list[Card] = gs.get("jokers", [])
    joker_slots: int = gs.get("joker_slots", 5)
    consumables: list[Card] = gs.get("consumables", [])
    consumable_slots: int = gs.get("consumable_slots", 2)

    if card_set == "Joker":
        is_negative = isinstance(card.edition, dict) and bool(card.edition.get("negative"))
        v[4] = float(len(jokers) < joker_slots or is_negative)
    elif card_set in ("Tarot", "Planet", "Spectral"):
        v[4] = float(len(consumables) < consumable_slots)
    else:
        v[4] = 1.0  # Vouchers, boosters, playing cards always "have slot"

    v[5] = _edition_idx(card.edition) / 4.0
    v[6] = float(card.eternal)
    v[7] = float(card.perishable)
    v[8] = float(card.rental)

    return v


def _compute_hand_analysis(
    hand: list[Card],
    jokers: list[Card],
    hand_levels: HandLevels | None,
) -> tuple[np.ndarray, str, set[int]]:
    """Compute hand type indicators, base score, and scoring card ids.

    Returns:
        hand_type_vec: (12,) binary array — which hand types the hand contains
        best_hand_name: name of the best detected hand type
        scoring_ids: set of id() values for cards in the best hand
    """
    hand_type_vec = np.zeros(NUM_HAND_TYPES, dtype=np.float32)
    scoring_ids: set[int] = set()
    best_hand_name = ""

    if not hand:
        return hand_type_vec, best_hand_name, scoring_ids

    # Get joker modifier flags
    flags = get_hand_eval_flags(jokers)

    # Detect best hand and all present hand types
    best_name, scoring_cards, full_results = get_best_hand(
        hand,
        four_fingers=flags["four_fingers"],
        shortcut=flags["shortcut"],
        smeared=flags["smeared"],
    )
    best_hand_name = best_name

    # Set binary indicators for all detected hand types
    for j, ht in enumerate(_HAND_TYPES):
        if full_results.get(ht.value):
            hand_type_vec[j] = 1.0

    # Track which cards are in the best hand
    scoring_ids = {id(c) for c in scoring_cards}

    return hand_type_vec, best_hand_name, scoring_ids


@dataclass
class _DrawAnalysis:
    """Result of draw proximity analysis."""

    flush_proximity: float  # max suit count / 5 (1.0 = have flush)
    straight_proximity: float  # max consecutive run / 5 (1.0 = have straight)
    best_flush_suit: str  # suit with most cards
    best_flush_count: int  # how many cards of that suit
    straight_gap_ranks: list[int]  # ranks needed to complete best straight


def _analyze_draws(hand: list[Card]) -> _DrawAnalysis:
    """Analyze flush and straight draw proximity with detail.

    Returns continuous proximity values and the specific suit/ranks
    needed to complete draws (for computing draw-out probabilities).
    """
    if not hand:
        return _DrawAnalysis(0.0, 0.0, "", 0, [])

    # Flush: count cards per suit, find best
    suit_counts: dict[str, int] = {}
    for card in hand:
        if card.base is None:
            continue
        suit = card.base.suit.value
        suit_counts[suit] = suit_counts.get(suit, 0) + 1

    best_suit = max(suit_counts, key=suit_counts.get, default="")  # type: ignore[arg-type]
    best_suit_count = suit_counts.get(best_suit, 0)
    flush_proximity = min(best_suit_count / 5.0, 1.0)

    # Straight: find longest consecutive run + gap ranks
    ranks: set[int] = set()
    for card in hand:
        if card.base is None:
            continue
        ranks.add(card.base.id)
    if 14 in ranks:
        ranks.add(1)  # Ace low

    best_run = 0
    best_run_start = 0
    current_run = 0
    current_start = 0
    sorted_ranks = sorted(ranks)

    for i, r in enumerate(sorted_ranks):
        if i == 0 or r != sorted_ranks[i - 1] + 1:
            current_run = 1
            current_start = r
        else:
            current_run += 1
        if current_run > best_run:
            best_run = current_run
            best_run_start = current_start

    straight_proximity = min(best_run / 5.0, 1.0)

    # Find which ranks would extend the best run to 5
    gap_ranks: list[int] = []
    if best_run >= 3:
        run_end = best_run_start + best_run - 1
        # Check ranks just below and above the run
        for r in range(best_run_start - 1, max(best_run_start - 3, 0), -1):
            if 2 <= r <= 14 and r not in ranks:
                gap_ranks.append(r)
        for r in range(run_end + 1, min(run_end + 3, 15)):
            if 2 <= r <= 14 and r not in ranks:
                gap_ranks.append(r)

    return _DrawAnalysis(
        flush_proximity=flush_proximity,
        straight_proximity=straight_proximity,
        best_flush_suit=best_suit,
        best_flush_count=best_suit_count,
        straight_gap_ranks=gap_ranks,
    )


def encode_global_context(gs: dict[str, Any]) -> np.ndarray:
    """Encode the fixed-size global context vector.

    Returns shape ``(D_GLOBAL,)`` float32 array (230 features).

    Layout:
        [0:6]      phase one-hot (6)
        [6:10]     blind_on_deck one-hot (4)
        [10:30]    scalar features (20)
        [30:90]    hand_levels 12 × 5 (60)
        [90:122]   vouchers_owned binary (32)
        [122:130]  blind_effect: boss, disabled, mult, debuff_suit(4), debuff_face (8)
        [130:133]  round_position one-hot: small/big/boss (3)
        [133:135]  round_progress: hands_played, discards_used (2)
        [135:159]  awarded_tags binary (24)
        [159:211]  discard_pile suit×rank histogram (52)
    """
    v = np.zeros(D_GLOBAL, dtype=np.float32)

    # Phase one-hot [0:6]
    phase = gs.get("phase", GamePhase.GAME_OVER)
    if isinstance(phase, str):
        try:
            phase = GamePhase(phase)
        except ValueError:
            phase = GamePhase.GAME_OVER
    phase_idx = _PHASE_IDX.get(phase, 5)
    v[phase_idx] = 1.0

    # Blind on deck one-hot [6:10]
    bod = gs.get("blind_on_deck")
    bod_idx = _BLIND_ON_DECK_IDX.get(bod, 0)
    v[6 + bod_idx] = 1.0

    # Scalar features [10:30] — inlined _log_scale for speed
    _ls = _log_scale  # local alias avoids global lookup per call
    rr = gs.get("round_resets", {})
    cr = gs.get("current_round", {})
    blind = gs.get("blind")
    blind_chips = getattr(blind, "chips", 0) if blind else 0
    chips = gs.get("chips", 0)
    blind_key = getattr(blind, "key", "") if blind else ""

    v[10] = rr.get("ante", 1) / 8.0
    v[11] = gs.get("round", 0) / 30.0
    v[12] = _ls(gs.get("dollars", 0))
    v[13] = cr.get("hands_left", 0) / 10.0
    v[14] = cr.get("discards_left", 0) / 10.0
    v[15] = gs.get("hand_size", 8) / 15.0
    v[16] = gs.get("joker_slots", 5) / 10.0
    v[17] = gs.get("consumable_slots", 2) / 5.0
    v[18] = _ls(blind_chips)
    v[19] = _ls(chips)
    v[20] = min(chips / max(blind_chips, 1), 10.0) / 10.0
    v[21] = cr.get("reroll_cost", 5) / 10.0
    v[22] = min(cr.get("free_rerolls", 0), 5) / 5.0
    v[23] = gs.get("interest_cap", 25) / 100.0
    v[24] = gs.get("discount_percent", 0) / 50.0
    v[25] = gs.get("skips", 0) / 10.0
    v[26] = center_key_id(blind_key) / max(NUM_CENTER_KEYS, 1)
    v[27] = _ls(len(gs.get("deck", [])))
    v[28] = _ls(len(gs.get("discard_pile", [])))
    v[29] = (
        float(bool(gs.get("four_fingers", 0)))
        + float(bool(gs.get("shortcut", 0))) * 2
        + float(bool(gs.get("smeared", 0))) * 4
        + float(bool(gs.get("splash", 0))) * 8
    ) / 15.0

    # Hand levels [30:90] — 12 hand types × 5 features
    hand_levels: HandLevels | None = gs.get("hand_levels")
    base_idx = 30
    for j, ht in enumerate(_HAND_TYPES):
        offset = base_idx + j * 5
        if hand_levels is not None:
            hs = hand_levels.get_state(ht)
            v[offset + 0] = hs.level / 20.0
            v[offset + 1] = _log_scale(hs.chips)
            v[offset + 2] = _log_scale(hs.mult)
            v[offset + 3] = _log_scale(hs.played)
            v[offset + 4] = float(hs.visible)

    # --- Vouchers owned [90:122] — 32-dim binary vector ---
    used_vouchers = gs.get("used_vouchers", {})
    vbase = _D_BASE  # 90
    for vk, vi in _VOUCHER_IDX.items():
        if used_vouchers.get(vk):
            v[vbase + vi] = 1.0

    # --- Blind effect features [122:130] — 8 dims ---
    bbase = vbase + NUM_VOUCHERS  # 122
    blind = gs.get("blind")
    if blind is not None:
        v[bbase + 0] = float(getattr(blind, "boss", False))
        v[bbase + 1] = float(getattr(blind, "disabled", False))
        v[bbase + 2] = getattr(blind, "mult", 1.0) / 4.0  # normalize (small=1, big=1.5, boss=2+)
        # Debuff category: suit one-hot (4 dims) + face debuff (1 dim)
        debuff_cfg = getattr(blind, "debuff_config", {}) or {}
        suit = debuff_cfg.get("suit")
        if suit and suit in _DEBUFF_SUIT_IDX:
            v[bbase + 3 + _DEBUFF_SUIT_IDX[suit]] = 1.0
        if debuff_cfg.get("is_face") == "face":
            v[bbase + 7] = 1.0

    # --- Round position [130:133] — 3-dim one-hot (which blind is current) ---
    rbase = bbase + 8  # 130
    rr = gs.get("round_resets", {})
    blind_states = rr.get("blind_states", {})
    for ri, bname in enumerate(("Small", "Big", "Boss")):
        state = blind_states.get(bname, "")
        if state in ("Current", "Select"):
            v[rbase + ri] = 1.0
            break  # only one can be active

    # --- Round progress [133:135] — hands_played, discards_used this round ---
    pbase = rbase + 3  # 133
    cr = gs.get("current_round", {})
    v[pbase + 0] = cr.get("hands_played", 0) / 10.0
    v[pbase + 1] = cr.get("discards_used", 0) / 10.0

    # --- Tags [135:159] — 24-dim binary vector of awarded tags ---
    tbase = pbase + 2  # 135
    awarded_tags = gs.get("awarded_tags", [])
    for tag_entry in awarded_tags:
        tag_key = tag_entry.get("key", "") if isinstance(tag_entry, dict) else ""
        ti = _TAG_IDX.get(tag_key)
        if ti is not None:
            v[tbase + ti] = 1.0

    # --- Discard pile histogram [159:211] — 52-dim (4 suits × 13 ranks) ---
    dbase = tbase + NUM_TAGS  # 159
    discard_pile: list = gs.get("discard_pile", [])
    for card in discard_pile:
        base = getattr(card, "base", None)
        if base is None:
            continue
        suit_str = base.suit.value if hasattr(base.suit, "value") else str(base.suit)
        rank_str = base.rank.value if hasattr(base.rank, "value") else str(base.rank)
        si = _SUIT_IDX.get(suit_str)
        ri_val = _RANK_IDX.get(rank_str)
        if si is not None and ri_val is not None:
            v[dbase + si * NUM_RANKS + ri_val] = 1.0

    # --- Strategic features [211:230] — 19 dims ---
    sbase = dbase + D_DISCARD_HISTOGRAM  # 211

    hand: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    hand_levels: HandLevels | None = gs.get("hand_levels")

    # Hand type indicators + best hand analysis [211:223]
    hand_type_vec, best_hand_name, _scoring_ids = _compute_hand_analysis(
        hand,
        jokers,
        hand_levels,
    )
    v[sbase : sbase + NUM_HAND_TYPES] = hand_type_vec

    # Base score estimate [223:226]
    score_base = sbase + NUM_HAND_TYPES  # 223
    if best_hand_name and hand_levels is not None:
        try:
            ht_enum = HandType(best_hand_name)
            base_data = HAND_BASE[ht_enum]
            hs = hand_levels.get_state(ht_enum)
            level = hs.level
            base_chips = base_data.chips_at(level)
            base_mult = base_data.mult_at(level)
            base_score = base_chips * base_mult
            v[score_base + 0] = _log_scale(base_chips)
            v[score_base + 1] = _log_scale(base_mult)
            v[score_base + 2] = _log_scale(base_score)
        except (ValueError, KeyError):
            pass

    # Score-to-blind ratio [226]
    ratio_base = score_base + 3  # 226
    base_score_val = 0.0
    if best_hand_name and hand_levels is not None:
        try:
            ht_enum = HandType(best_hand_name)
            bd = HAND_BASE[ht_enum]
            hs = hand_levels.get_state(ht_enum)
            base_score_val = bd.chips_at(hs.level) * bd.mult_at(hs.level)
        except (ValueError, KeyError):
            pass
    if base_score_val > 0 and blind_chips > 0:
        v[ratio_base] = min(math.log1p(base_score_val) / math.log1p(blind_chips), 2.0) / 2.0

    # Draw proximity [227:229] — continuous, not binary
    draw_base = ratio_base + 1  # 227
    draws = _analyze_draws(hand)
    v[draw_base + 0] = draws.flush_proximity
    v[draw_base + 1] = draws.straight_proximity

    # Interest earned [229]
    int_base = draw_base + 2  # 229
    dollars = gs.get("dollars", 0)
    interest_cap = gs.get("interest_cap", 25)
    max_interest = interest_cap // 5
    interest = min(max(dollars, 0) // 5, max_interest)
    v[int_base] = interest / max(max_interest, 1)

    # --- New features [230:235] ---

    # Flush draw outs [230] — P(drawing suit-completing card) from remaining deck
    outs_base = int_base + 1  # 230
    deck: list = gs.get("deck", [])
    deck_size = len(deck)
    if draws.best_flush_count < 5 and draws.best_flush_suit and deck_size > 0:
        suited_in_deck = sum(
            1
            for c in deck
            if getattr(c, "base", None) is not None and c.base.suit.value == draws.best_flush_suit
        )
        v[outs_base] = suited_in_deck / deck_size

    # Straight draw outs [231] — P(drawing rank-completing card) from remaining deck
    if draws.straight_gap_ranks and deck_size > 0:
        gap_set = set(draws.straight_gap_ranks)
        matching_in_deck = sum(
            1 for c in deck if getattr(c, "base", None) is not None and c.base.id in gap_set
        )
        v[outs_base + 1] = matching_in_deck / deck_size

    # Boss blind debuff impact [232] — fraction of hand cards debuffed
    debuff_base = outs_base + 2  # 232
    if hand:
        n_debuffed = sum(1 for c in hand if c.debuff)
        v[debuff_base] = n_debuffed / len(hand)

    # Spendable above interest [233] — money that can be spent without losing interest
    spend_base = debuff_base + 1  # 233
    interest_floor = interest * 5  # dollars needed to maintain current interest
    spendable = max(0, dollars - interest_floor)
    v[spend_base] = _log_scale(spendable)

    # Urgency [234] — blind_remaining / (base_score × hands_left)
    # High values = in trouble, need joker help or better hands
    urg_base = spend_base + 1  # 234
    hands_left = gs.get("current_round", {}).get("hands_left", 0)
    blind_remaining = max(blind_chips - chips, 0)
    if base_score_val > 0 and hands_left > 0:
        urgency = blind_remaining / (base_score_val * hands_left)
        v[urg_base] = min(urgency, 5.0) / 5.0

    return v


# ---------------------------------------------------------------------------
# Batch encoding functions (hot path optimization)
# ---------------------------------------------------------------------------

# Pre-allocated buffers keyed by max entity count, reused across calls.
# Thread-local would be needed for multi-threaded use, but RL envs are
# single-threaded per worker.
_HAND_BUF: np.ndarray | None = None
_JOKER_BUF: np.ndarray | None = None

_EMPTY_PLAYING = np.zeros((0, D_PLAYING_CARD), dtype=np.float32)
_EMPTY_JOKER = np.zeros((0, D_JOKER), dtype=np.float32)
_EMPTY_CONSUMABLE = np.zeros((0, D_CONSUMABLE), dtype=np.float32)
_EMPTY_SHOP = np.zeros((0, D_SHOP), dtype=np.float32)


def _get_hand_buf(n: int) -> np.ndarray:
    """Return a pre-allocated (n, D_PLAYING_CARD) buffer, reusing when possible."""
    global _HAND_BUF
    if _HAND_BUF is None or _HAND_BUF.shape[0] < n:
        _HAND_BUF = np.zeros((max(n, 16), D_PLAYING_CARD), dtype=np.float32)
    buf = _HAND_BUF[:n]
    buf[:] = 0.0
    return buf


def _get_joker_buf(n: int) -> np.ndarray:
    """Return a pre-allocated (n, D_JOKER) buffer, reusing when possible."""
    global _JOKER_BUF
    if _JOKER_BUF is None or _JOKER_BUF.shape[0] < n:
        _JOKER_BUF = np.zeros((max(n, 8), D_JOKER), dtype=np.float32)
    buf = _JOKER_BUF[:n]
    buf[:] = 0.0
    return buf


def encode_playing_cards_batch(
    cards: list[Card],
    gs: dict[str, Any],
) -> np.ndarray:
    """Encode multiple playing cards into a single array.

    Returns shape ``(len(cards), D_PLAYING_CARD)`` float32 array.
    Uses pre-allocated buffer to avoid per-call allocation.
    """
    n = len(cards)
    if n == 0:
        return _EMPTY_PLAYING
    buf = _get_hand_buf(n)
    splash = bool(gs.get("splash", 0))

    # Compute which cards belong to the best detected hand
    jokers: list[Card] = gs.get("jokers", [])
    _, _, scoring_ids = _compute_hand_analysis(cards, jokers, gs.get("hand_levels"))

    for i, card in enumerate(cards):
        row = buf[i]  # already zeroed
        face_down = card.facing == "back"
        if face_down or card.base is None:
            row[7] = 1.0
            row[12] = i / 20.0
            continue

        rank_str = card.base.rank.value
        suit_str = card.base.suit.value

        row[0] = card.base.id / 14.0
        row[1] = _SUIT_IDX.get(suit_str, 0) / 3.0
        row[2] = _RANK_CHIPS.get(rank_str, 0) / 11.0
        row[3] = _ENHANCEMENT_IDX.get(card.center_key, 0) / 8.0
        row[4] = _edition_idx(card.edition) / 4.0
        row[5] = _SEAL_IDX.get(card.seal, 0) / 4.0
        row[6] = float(card.debuff)
        # row[7] = 0.0 already
        row[8] = float(card.base.id in (11, 12, 13))
        row[9] = float(splash or not card.debuff)
        row[10] = _log_scale(card.ability.get("bonus", 0) + card.ability.get("perma_bonus", 0))
        row[11] = _log_scale(card.base.times_played)
        row[12] = i / 20.0
        row[13] = float(id(card) in scoring_ids)

    # Return a copy so the buffer can be reused next call
    return buf.copy()


def encode_jokers_batch(
    jokers: list[Card],
    gs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Encode multiple jokers into a single array.

    Parameters
    ----------
    jokers : list[Card]
        Joker cards to encode.
    gs : dict, optional
        Game state dict (reserved for future use).

    Returns shape ``(len(jokers), D_JOKER)`` float32 array.
    """
    n = len(jokers)
    if n == 0:
        return _EMPTY_JOKER
    buf = _get_joker_buf(n)
    nck = max(NUM_CENTER_KEYS, 1)

    for i, card in enumerate(jokers):
        row = buf[i]
        row[0] = center_key_id(card.center_key) / nck
        row[1] = min(card.base_cost / 20.0, 1.0) if card.base_cost else 0.0
        row[2] = _edition_idx(card.edition) / 4.0
        row[3] = _log_scale(card.sell_cost)
        row[4] = float(card.eternal)
        row[5] = float(card.perishable)
        row[6] = card.perish_tally / 5.0 if card.perishable else 1.0
        row[7] = float(card.rental)
        row[8] = float(card.debuff)
        row[9] = i / 20.0
        row[10] = _log_scale(card.ability.get("mult", 0) + card.ability.get("t_mult", 0))
        row[11] = card.ability.get("x_mult", 1.0)
        row[12] = _log_scale(card.ability.get("t_chips", 0) + card.ability.get("bonus", 0))
        extra = card.ability.get("extra")
        if isinstance(extra, (int, float)):
            row[13] = _log_scale(extra)
        elif isinstance(extra, dict):
            total = 0.0
            for val in extra.values():
                if isinstance(val, (int, float)):
                    total += val
            row[13] = _log_scale(total)
        row[14] = float(not card.debuff)

    return buf.copy()


# ---------------------------------------------------------------------------
# Observation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """Entity-based observation — variable-length entity lists + fixed global.

    Each entity array has shape ``(N, D)`` where N varies per timestep.
    Empty areas produce shape ``(0, D)`` arrays.  All arrays are float32.
    """

    global_context: np.ndarray  # (D_GLOBAL,)
    hand_cards: np.ndarray  # (N_hand, D_PLAYING_CARD)
    jokers: np.ndarray  # (N_joker, D_JOKER)
    consumables: np.ndarray  # (N_cons, D_CONSUMABLE)
    shop_cards: np.ndarray  # (N_shop, D_SHOP)
    pack_cards: np.ndarray  # (N_pack, D_PLAYING_CARD)

    def to_game_observation(self) -> GameObservation:
        """Convert to a game-agnostic :class:`GameObservation`."""
        return GameObservation(
            global_context=self.global_context,
            entities={
                "hand_card": self.hand_cards,
                "joker": self.jokers,
                "consumable": self.consumables,
                "shop_item": self.shop_cards,
                "pack_card": self.pack_cards,
            },
        )


# ---------------------------------------------------------------------------
# Top-level encoder
# ---------------------------------------------------------------------------


def encode_observation(gs: dict[str, Any]) -> Observation:
    """Encode a full game state into an :class:`Observation`.

    Parameters
    ----------
    gs : dict
        The engine game_state dict (from ``DirectAdapter.raw_state`` or
        equivalent).

    Returns
    -------
    Observation
        Entity-based observation with variable-length arrays.
    """
    # Global context
    global_ctx = encode_global_context(gs)

    # Hand cards — batch encode
    hand: list[Card] = gs.get("hand", [])
    hand_arr = encode_playing_cards_batch(hand, gs)

    # Jokers — batch encode
    jokers: list[Card] = gs.get("jokers", [])
    joker_arr = encode_jokers_batch(jokers, gs)

    # Consumables (small count, keep per-card)
    consumables: list[Card] = gs.get("consumables", [])
    if consumables:
        cons_arr = np.stack([encode_consumable(c, gs) for c in consumables])
    else:
        cons_arr = _EMPTY_CONSUMABLE

    # Shop cards (combine shop_cards + shop_vouchers + shop_boosters)
    shop_items: list[Card] = (
        gs.get("shop_cards", []) + gs.get("shop_vouchers", []) + gs.get("shop_boosters", [])
    )
    if shop_items:
        shop_arr = np.stack([encode_shop_item(c, gs) for c in shop_items])
    else:
        shop_arr = _EMPTY_SHOP

    # Pack cards — batch encode
    pack_cards: list[Card] = gs.get("pack_cards", [])
    pack_arr = encode_playing_cards_batch(pack_cards, gs)

    return Observation(
        global_context=global_ctx,
        hand_cards=hand_arr,
        jokers=joker_arr,
        consumables=cons_arr,
        shop_cards=shop_arr,
        pack_cards=pack_arr,
    )
