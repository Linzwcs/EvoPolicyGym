"""Poker hand detection functions.

Pure functions that take a list of cards and return which cards form each
component.  Ported from ``misc_functions.lua:376-621``.

These are the building blocks used by ``evaluate_poker_hand`` (ported
separately) to determine the best hand from played cards.

Joker modifiers that affect detection:
  - **Four Fingers**: flush/straight need only 4 cards instead of 5
  - **Shortcut**: straights allow 1-rank gaps
  - **Smeared Joker**: Hearts=Diamonds and Spades=Clubs for suit checks
  - **Wild Card** (enhancement): counts as every suit for flush detection
  - **Stone Card** (enhancement): excluded from flush suit matching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


def is_suit(
    card: Card,
    suit: str,
    *,
    flush_calc: bool = False,
    smeared: bool = False,
) -> bool:
    """Check if *card* matches *suit*.

    Delegates to ``Card.is_suit`` (card.lua:4064).  Kept as a module-level
    function for backward compatibility with existing tests.
    """
    return card.is_suit(suit, flush_calc=flush_calc, smeared=smeared)


def get_flush(
    hand: list[Card],
    *,
    four_fingers: bool = False,
    smeared: bool = False,
) -> list[list[Card]]:
    """Detect a flush in *hand*, matching ``get_flush`` (misc_functions.lua:522).

    Returns a list containing one element (the list of flush cards) if found,
    or an empty list if no flush.  Checks suits in order: Spades, Hearts,
    Clubs, Diamonds — returns on the **first** qualifying suit.

    Args:
        four_fingers: If True, only 4 cards needed (not 5).
        smeared: If True, Hearts/Diamonds and Spades/Clubs are interchangeable.
    """
    threshold = 4 if four_fingers else 5

    if len(hand) > 5 or len(hand) < threshold:
        return []

    suits = ["Spades", "Hearts", "Clubs", "Diamonds"]

    for suit in suits:
        t: list[Card] = []
        for card in hand:
            if is_suit(card, suit, flush_calc=True, smeared=smeared):
                t.append(card)
        if len(t) >= threshold:
            return [t]

    return []


def get_straight(
    hand: list[Card],
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
) -> list[list[Card]]:
    """Detect a straight in *hand*, matching ``get_straight`` (misc_functions.lua:548).

    Returns a list containing one element (the list of straight cards) if
    found, or an empty list if no straight.

    Ace can be high (10-J-Q-K-A) or low (A-2-3-4-5) but does NOT wrap
    (Q-K-A-2-3 is invalid).

    Args:
        four_fingers: If True, only 4 consecutive ranks needed.
        shortcut: If True, one rank gap is allowed (e.g. 3-4-6-7-8).
    """
    threshold = 4 if four_fingers else 5

    if len(hand) > 5 or len(hand) < threshold:
        return []

    # Build IDS: rank_id → list of cards at that rank
    # IDs range from 2-14 (Ace=14)
    ids: dict[int, list[Card]] = {}
    for card in hand:
        card_id = card.get_id()
        if 1 < card_id < 15:
            if card_id in ids:
                ids[card_id].append(card)
            else:
                ids[card_id] = [card]

    straight_length = 0
    straight = False
    skipped_rank = False
    t: list[Card] = []

    # j goes from 1 to 14
    # When j==1, check ids[14] (Ace-low: A counted first for A-2-3-4-5)
    # When j==2..14, check ids[j]
    for j in range(1, 15):
        check_id = 14 if j == 1 else j

        if check_id in ids:
            straight_length += 1
            skipped_rank = False
            t.extend(ids[check_id])
        elif shortcut and not skipped_rank and j != 14:
            # Allow one gap (but not at the end, j==14)
            skipped_rank = True
        else:
            straight_length = 0
            skipped_rank = False
            if not straight:
                t = []
            if straight:
                break

        if straight_length >= threshold:
            straight = True

    if not straight:
        return []
    return [t]


def get_x_same(num: int, hand: list[Card]) -> list[list[Card]]:
    """Find all groups of exactly *num* cards sharing the same rank.

    Matches ``get_X_same`` (misc_functions.lua:592).

    Returns groups ordered by rank descending (highest first).
    Each group contains exactly *num* cards.
    """
    # vals[id] = list of cards with that id, only if count == num
    vals: dict[int, list[Card]] = {}

    for i in range(len(hand) - 1, -1, -1):
        curr = [hand[i]]
        card_id = hand[i].get_id()
        for j in range(len(hand)):
            if hand[i].get_id() == hand[j].get_id() and i != j:
                curr.append(hand[j])
        if len(curr) == num:
            vals[card_id] = curr

    # Return in descending rank order
    ret: list[list[Card]] = []
    for rank_id in range(14, 0, -1):
        if rank_id in vals:
            ret.append(vals[rank_id])
    return ret


def get_highest(hand: list[Card]) -> list[list[Card]]:
    """Return the single highest-nominal card, matching ``get_highest`` (misc_functions.lua:613).

    Returns ``[[highest_card]]`` or ``[]`` if hand is empty.
    """
    if not hand:
        return []

    from jackdaw.engine.card_area import _card_nominal

    highest = max(hand, key=_card_nominal)
    return [[highest]]


# ---------------------------------------------------------------------------
# Joker modifier flag extraction
# ---------------------------------------------------------------------------

# Mapping from P_CENTERS key → flag name for meta jokers that affect detection
_META_JOKER_FLAGS: dict[str, str] = {
    "j_four_fingers": "four_fingers",
    "j_shortcut": "shortcut",
    "j_smeared": "smeared",
    "j_splash": "splash",
    "j_pareidolia": "pareidolia",
}


def find_joker(name: str, jokers: list[Card], *, non_debuff: bool = False) -> list[Card]:
    """Find jokers by ability name, matching ``find_joker`` (misc_functions.lua:903).

    By default, **debuffed jokers are excluded** — the source's default call
    ``find_joker('name')`` passes ``non_debuff=nil``, and the condition
    ``(non_debuff or not v.debuff)`` evaluates to ``not v.debuff`` when
    ``non_debuff`` is falsy.

    Args:
        name: The ``ability.name`` to match (e.g. ``"Four Fingers"``).
        jokers: List of joker Card objects to search.
        non_debuff: If True, include debuffed jokers too.

    Returns:
        List of matching (non-debuffed by default) joker Cards.
    """
    return [j for j in jokers if j.ability.get("name") == name and (non_debuff or not j.debuff)]


def get_hand_eval_flags(jokers: list[Card]) -> dict[str, bool]:
    """Extract hand evaluation modifier flags from active jokers.

    These jokers modify hand **detection**, not scoring.  They are checked
    via ``find_joker`` (misc_functions.lua:903), which by default excludes
    debuffed jokers — so a debuffed Four Fingers does NOT enable 4-card
    flushes.

    Flags:
        - ``four_fingers``: flush/straight need only 4 cards
        - ``shortcut``: straights allow 1-rank gaps
        - ``smeared``: Hearts=Diamonds, Spades=Clubs for suit checks
        - ``splash``: all played cards become scoring cards (applied in
          evaluate_play, not in hand detection itself)
        - ``pareidolia``: all cards count as face cards (affects is_face
          checks, not hand detection directly)

    Args:
        jokers: List of joker Card objects (typically ``G.jokers.cards``).

    Returns:
        Dict of flag name → bool.
    """
    flags = {
        "four_fingers": False,
        "shortcut": False,
        "smeared": False,
        "splash": False,
        "pareidolia": False,
    }
    for j in jokers:
        if j.debuff:
            continue
        flag = _META_JOKER_FLAGS.get(j.center_key)
        if flag:
            flags[flag] = True
    return flags


# ---------------------------------------------------------------------------
# evaluate_poker_hand — master detection (misc_functions.lua:376)
# ---------------------------------------------------------------------------

# Type alias for the results dict
HandResults = dict[str, list[list["Card"]]]


def evaluate_poker_hand(
    hand: list[Card],
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
) -> HandResults:
    """Detect all poker hands present in *hand*.

    Matches ``evaluate_poker_hand`` (misc_functions.lua:376-519).

    Returns a dict mapping hand name strings to lists of card groups.
    Multiple hands can be populated simultaneously (e.g. a Full House
    also populates Three of a Kind and Pair).

    Downward propagation (lines 507-517): Five of a Kind also populates
    Four/Three/Pair.  Four of a Kind also populates Three/Pair.  Three
    of a Kind also populates Pair.  This ensures jokers like The Duo
    that check ``poker_hands['Pair']`` work regardless of the actual
    detected hand.

    Args:
        four_fingers: If True, flush/straight need only 4 cards.
        shortcut: If True, straights allow 1-rank gaps.
        smeared: If True, Hearts/Diamonds and Spades/Clubs are
            interchangeable for flushes.
    """
    results: HandResults = {
        "Flush Five": [],
        "Flush House": [],
        "Five of a Kind": [],
        "Straight Flush": [],
        "Four of a Kind": [],
        "Full House": [],
        "Flush": [],
        "Straight": [],
        "Three of a Kind": [],
        "Two Pair": [],
        "Pair": [],
        "High Card": [],
    }

    # Compute component parts
    _5 = get_x_same(5, hand)
    _4 = get_x_same(4, hand)
    _3 = get_x_same(3, hand)
    _2 = get_x_same(2, hand)
    _flush = get_flush(hand, four_fingers=four_fingers, smeared=smeared)
    _straight = get_straight(hand, four_fingers=four_fingers, shortcut=shortcut)
    _highest = get_highest(hand)

    # Check in priority order — populate all that match

    # Flush Five: 5-of-a-kind AND flush
    if _5 and _flush:
        results["Flush Five"] = _5

    # Flush House: 3-of-a-kind AND pair AND flush
    if _3 and _2 and _flush:
        fh_hand = list(_3[0]) + list(_2[0])
        results["Flush House"] = [fh_hand]

    # Five of a Kind
    if _5:
        results["Five of a Kind"] = _5

    # Straight Flush: flush AND straight (merge cards from both)
    if _flush and _straight:
        # Start with flush cards, add straight cards not already in flush
        flush_set = set(id(c) for c in _flush[0])
        merged = list(_flush[0])
        for c in _straight[0]:
            if id(c) not in flush_set:
                merged.append(c)
        results["Straight Flush"] = [merged]

    # Four of a Kind
    if _4:
        results["Four of a Kind"] = _4

    # Full House: 3-of-a-kind AND pair
    if _3 and _2:
        fh_hand = list(_3[0]) + list(_2[0])
        results["Full House"] = [fh_hand]

    # Flush
    if _flush:
        results["Flush"] = _flush

    # Straight
    if _straight:
        results["Straight"] = _straight

    # Three of a Kind
    if _3:
        results["Three of a Kind"] = _3

    # Two Pair: 2 pairs, OR 1 triple + 1 pair (the triple counts as a "pair")
    if len(_2) >= 2 or (len(_3) == 1 and len(_2) == 1):
        if len(_2) >= 2:
            tp_hand = list(_2[0]) + list(_2[1])
        else:
            # _3 exists but only 1 _2 → use _3[0] as the second "pair"
            tp_hand = list(_2[0]) + list(_3[0])
        results["Two Pair"] = [tp_hand]

    # Pair
    if _2:
        results["Pair"] = _2

    # High Card
    if _highest:
        results["High Card"] = _highest

    # -- Downward propagation (misc_functions.lua:507-517) --
    # Five of a Kind → Four/Three/Pair
    if results["Five of a Kind"]:
        results["Four of a Kind"] = results["Five of a Kind"]
    # Four of a Kind → Three/Pair
    if results["Four of a Kind"]:
        results["Three of a Kind"] = results["Four of a Kind"]
    # Three of a Kind → Pair
    if results["Three of a Kind"]:
        results["Pair"] = results["Three of a Kind"]

    return results


def get_best_hand(
    hand: list[Card],
    *,
    four_fingers: bool = False,
    shortcut: bool = False,
    smeared: bool = False,
) -> tuple[str, list[Card], HandResults]:
    """Determine the best poker hand and its scoring cards.

    Matches ``G.FUNCS.get_poker_hand_info`` (state_events.lua:540).

    Returns:
        ``(hand_name, scoring_cards, full_results)`` where:
        - ``hand_name`` is the detected hand type string (e.g. ``"Full House"``)
        - ``scoring_cards`` is the list of cards forming that hand
        - ``full_results`` is the complete dict from ``evaluate_poker_hand``
    """
    from jackdaw.engine.data.hands import HAND_ORDER

    results = evaluate_poker_hand(
        hand,
        four_fingers=four_fingers,
        shortcut=shortcut,
        smeared=smeared,
    )

    # Walk priority order — first non-empty match wins
    for ht in HAND_ORDER:
        if results[ht.value]:
            return ht.value, results[ht.value][0], results

    return "NULL", [], results


# ---------------------------------------------------------------------------
# HandEvalResult — output of the complete evaluation pipeline
# ---------------------------------------------------------------------------


@dataclass
class HandEvalResult:
    """Result of the complete hand evaluation pipeline.

    Contains everything the downstream scoring pipeline (Phase 3+) needs.
    """

    detected_hand: str
    """The best hand type detected (e.g. ``"Full House"``), or ``"NULL"``."""

    poker_hands: HandResults
    """Full detection results for all 12 hand types.  Used by jokers
    that check sub-hand presence (e.g. The Duo checks for Pair)."""

    scoring_cards: list[Card]
    """Cards that will score, ordered left-to-right.  Includes Stone Card
    augmentation and Splash expansion."""

    all_played: list[Card]
    """All cards that were played (``G.play.cards``), for context."""


# ---------------------------------------------------------------------------
# evaluate_hand — complete pipeline entry point
# ---------------------------------------------------------------------------


def evaluate_hand(
    played_cards: list[Card],
    jokers: list[Card] | None = None,
) -> HandEvalResult:
    """Complete hand evaluation pipeline.

    Ties together modifier flag extraction, poker hand detection, scoring
    card augmentation (Splash, Stone), and result packaging.

    Replaces the source's ``get_poker_hand_info`` + scoring hand
    augmentation from ``evaluate_play`` (state_events.lua:571-600).

    Steps:
        1. Extract modifier flags from active jokers
        2. Run ``evaluate_poker_hand`` with detection flags
        3. Determine the detected (best) hand type via priority walk
        4. Augment scoring cards: Splash adds all played cards,
           Stone Cards are added as "pures" if not already included
        5. Return ``HandEvalResult``

    Args:
        played_cards: Cards the player chose to play (1-5 cards).
        jokers: Active joker cards (for modifier flag extraction).
            Pass ``None`` or ``[]`` if no jokers.
    """
    from jackdaw.engine.data.hands import HAND_ORDER

    joker_list = jokers or []
    flags = get_hand_eval_flags(joker_list)

    # Detection flags for evaluate_poker_hand (only 3 of 5 flags apply)
    results = evaluate_poker_hand(
        played_cards,
        four_fingers=flags["four_fingers"],
        shortcut=flags["shortcut"],
        smeared=flags["smeared"],
    )

    # Determine best hand
    detected_hand = "NULL"
    scoring_cards: list[Card] = []
    for ht in HAND_ORDER:
        if results[ht.value]:
            detected_hand = ht.value
            scoring_cards = list(results[ht.value][0])
            break

    # -- Augment scoring cards (state_events.lua:580-600) --

    # Splash joker: ALL played cards become scoring cards
    if flags["splash"]:
        scoring_ids = {id(c) for c in scoring_cards}
        for c in played_cards:
            if id(c) not in scoring_ids:
                scoring_cards.append(c)

    # Stone Cards: add as "pures" if not already in scoring hand
    scoring_ids = {id(c) for c in scoring_cards}
    for c in played_cards:
        if id(c) not in scoring_ids and c.ability.get("effect") == "Stone Card":
            scoring_cards.append(c)

    # Sort scoring cards by position (left-to-right = by index in played_cards)
    played_order = {id(c): i for i, c in enumerate(played_cards)}
    scoring_cards.sort(key=lambda c: played_order.get(id(c), 999))

    return HandEvalResult(
        detected_hand=detected_hand,
        poker_hands=results,
        scoring_cards=scoring_cards,
        all_played=list(played_cards),
    )
