"""Pack card generation for Balatro.

Ports the inner card-generation loop of ``Card:open()`` from
``card.lua:1728-1781`` for the five booster pack types.

Source references
-----------------
- card.lua:1728 — ``Card:open()`` inner loop
- common_events.lua:2082 — ``create_card`` (Base/Enhanced front selection
  at line 2124)
- common_events.lua:2055 — ``poll_edition``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jackdaw.engine.card_factory import create_card, create_playing_card
from jackdaw.engine.card_utils import poll_edition
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import BOOSTERS, CENTER_POOLS, PLANETS, PLAYING_CARDS

if TYPE_CHECKING:
    from jackdaw.engine.card import Card
    from jackdaw.engine.rng import PseudoRandom


def generate_pack_cards(
    pack_key: str,
    rng: PseudoRandom,
    ante: int,
    game_state: dict[str, Any],
) -> tuple[list[Card], int]:
    """Generate the cards shown when opening a booster pack.

    Mirrors the inner loop of ``Card:open()`` in ``card.lua:1728-1781``.

    Parameters
    ----------
    pack_key:
        Booster key from BOOSTERS (e.g. ``"p_arcana_normal_1"``).
    rng:
        Live PseudoRandom instance.
    ante:
        Current ante number.
    game_state:
        Dict of game-state values.  Relevant keys beyond those forwarded to
        :func:`~jackdaw.engine.card_factory.create_card`:

        ``has_omen_globe`` (bool):
            Omen Globe voucher is active — enables 20% Spectral substitution
            in Arcana packs.

        ``has_telescope`` (bool):
            Telescope voucher is active — forces the first Celestial pack card
            to the planet matching ``most_played_hand``.

        ``most_played_hand`` (str | None):
            The hand type name (e.g. ``"Flush"``) with the most plays this
            run.  Required for the Telescope effect.

        ``edition_rate`` (float, default ``1.0``):
            Edition rate multiplier forwarded to :func:`poll_edition` for
            Standard pack edition rolls.

    Returns
    -------
    tuple[list[Card], int]
        ``(cards, choose)`` where *cards* is the list of generated cards
        and *choose* is how many the player may select from the pack.
    """
    proto = BOOSTERS[pack_key]
    kind = proto.kind
    extra: int = proto.config.get("extra", 1)
    choose: int = proto.config.get("choose", 1)
    gs = game_state or {}

    cards: list[Card] = []
    # Track keys added during this pack generation so we can clean up after.
    # In Lua, Card:set_ability (card.lua:349-354) adds every created card's
    # center key to G.GAME.used_jokers, preventing duplicates within a pack.
    # The keys are later removed when unpicked cards are destroyed
    # (card.lua:4741-4748).  We replicate this by temporarily adding keys
    # during generation and removing them after.
    _pack_added_keys: list[str] = []
    for i in range(extra):
        if kind == "Arcana":
            card = _gen_arcana(rng, ante, gs)
        elif kind == "Celestial":
            card = _gen_celestial(rng, ante, gs, i)
        elif kind == "Spectral":
            card = _gen_spectral(rng, ante, gs)
        elif kind == "Standard":
            card = _gen_standard(rng, ante, gs)
        elif kind == "Buffoon":
            card = _gen_buffoon(rng, ante, gs)
        else:
            raise ValueError(f"Unknown pack kind: {kind!r}")
        cards.append(card)

        if "used_jokers" in gs and card.center_key != "c_base":
            if card.center_key not in gs["used_jokers"]:
                gs["used_jokers"][card.center_key] = True
                _pack_added_keys.append(card.center_key)

    # Clean up temporarily added keys (will be re-added by pick logic
    # for whichever card the player selects)
    for k in _pack_added_keys:
        del gs["used_jokers"][k]

    return cards, choose


# ---------------------------------------------------------------------------
# Per-type generators
# ---------------------------------------------------------------------------


def _gen_arcana(rng: PseudoRandom, ante: int, gs: dict) -> Card:
    """Generate one Arcana (Tarot/Spectral) pack card.

    Source: card.lua:1730-1735.  The Omen Globe voucher gives each slot a
    20% chance to become a Spectral instead of a Tarot.  The RNG stream
    ``'omen_globe'`` is advanced **only** when the voucher is active
    (Lua short-circuit evaluation).
    """
    if gs.get("has_omen_globe") and rng.random("omen_globe") > 0.8:
        return create_card(
            "Spectral", rng, ante, area="pack", append="ar2", soulable=True, game_state=gs
        )
    return create_card("Tarot", rng, ante, area="pack", append="ar1", soulable=True, game_state=gs)


def _gen_celestial(rng: PseudoRandom, ante: int, gs: dict, slot_idx: int) -> Card:
    """Generate one Celestial (Planet) pack card.

    Source: card.lua:1736-1755.  The Telescope voucher forces the first
    card (slot 0) to be the planet matching the most-played hand type.
    """
    forced_key: str | None = None
    if gs.get("has_telescope") and slot_idx == 0:
        most_played = gs.get("most_played_hand")
        if most_played:
            for planet_key, planet_proto in PLANETS.items():
                if planet_proto.config.get("hand_type") == most_played:
                    forced_key = planet_key
                    break

    return create_card(
        "Planet",
        rng,
        ante,
        area="pack",
        append="pl1",
        soulable=True,
        forced_key=forced_key,
        game_state=gs,
    )


def _gen_spectral(rng: PseudoRandom, ante: int, gs: dict) -> Card:
    """Generate one Spectral pack card.  Source: card.lua:1756-1757."""
    return create_card(
        "Spectral", rng, ante, area="pack", append="spe", soulable=True, game_state=gs
    )


def _gen_buffoon(rng: PseudoRandom, ante: int, gs: dict) -> Card:
    """Generate one Buffoon (Joker) pack card.  Source: card.lua:1773-1774."""
    return create_card("Joker", rng, ante, area="pack", append="buf", soulable=True, game_state=gs)


def _gen_standard(rng: PseudoRandom, ante: int, gs: dict) -> Card:
    """Generate one Standard pack playing card.

    Source: card.lua:1758-1772 + common_events.lua:2103-2124.

    Pipeline
    --------
    1. **Type roll** — ``pseudorandom(pseudoseed('stdset'+ante)) > 0.6``
       selects *Enhanced* or *Base*.
    2. **Front selection** — playing card (suit + rank) picked from P_CARDS
       using ``pseudoseed('front'+'sta'+ante)`` (common_events.lua:2124).
    3. **Enhancement** — if *Enhanced*, one enhancement key picked from the
       ``Enhanced`` center pool using ``pseudoseed('Enhancedsta')``.
    4. **Edition** — ``poll_edition('standard_edition'+ante, mod=2, no_neg=True)``.
    5. **Seal** — ``pseudorandom(pseudoseed('stdseal'+ante)) > 0.8`` triggers
       a seal; type determined by ``pseudoseed('stdsealtype'+ante)`` split into
       four equal quarters (Red/Blue/Gold/Purple).
    """
    is_enhanced = rng.random("stdset" + str(ante)) > 0.6

    # Pick the playing card (front) from P_CARDS
    pc_proto, _ = rng.element(PLAYING_CARDS, rng.seed("front" + "sta" + str(ante)))
    suit = Suit(pc_proto.suit)
    rank = Rank(pc_proto.rank)

    # Enhancement selection (only when Enhanced type)
    if is_enhanced:
        enhancement_key, _ = rng.element(
            CENTER_POOLS["Enhanced"],
            rng.seed("Enhancedsta"),
        )
    else:
        enhancement_key = "c_base"

    card = create_playing_card(suit, rank, enhancement=enhancement_key)

    # Edition
    edition = poll_edition(
        "standard_edition" + str(ante),
        rng,
        rate=gs.get("edition_rate", 1.0),
        mod=2,
        no_neg=True,
    )
    if edition:
        card.set_edition(edition)

    # Seal
    if rng.random("stdseal" + str(ante)) > 0.8:
        seal_roll = rng.random("stdsealtype" + str(ante))
        if seal_roll > 0.75:
            card.set_seal("Red")
        elif seal_roll > 0.5:
            card.set_seal("Blue")
        elif seal_roll > 0.25:
            card.set_seal("Gold")
        else:
            card.set_seal("Purple")

    return card
