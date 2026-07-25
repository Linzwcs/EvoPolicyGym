"""Canonical game_state schema.

The game state is a plain ``dict[str, Any]`` threaded through
:func:`~jackdaw.engine.run_init.initialize_run` (creation) and
:func:`~jackdaw.engine.game.step` (mutation).  This module documents
every key, its type, and which module is responsible for it.

This is documentation only — no runtime enforcement.  The dict approach
was chosen for speed (no dataclass overhead on hot paths) and flexibility
(easy to extend without migration).

Lifecycle legend for the **set by** column:

- **init** = :func:`~jackdaw.engine.run_init.init_game_object`
- **run_init** = :func:`~jackdaw.engine.run_init.initialize_run`
- **stakes** = :func:`~jackdaw.engine.stakes.apply_stake_modifiers`
- **back** = :meth:`~jackdaw.engine.back.Back.apply_to_run`
- **challenge** = :func:`~jackdaw.engine.challenges.apply_challenge`
- **voucher** = :func:`~jackdaw.engine.vouchers.apply_voucher`
- **game** = :func:`~jackdaw.engine.game.step` (various handlers)
- **runner** = :func:`~jackdaw.engine.runner.simulate_run`
- **round** = :func:`~jackdaw.engine.run_init.start_round`
- **scoring** = :func:`~jackdaw.engine.scoring.score_hand`
- **economy** = :func:`~jackdaw.engine.economy.calculate_round_earnings`
- **shop** = shop population / reroll helpers in ``game.py``
- **tags** = :func:`~jackdaw.engine.tags.assign_ante_blinds`
- **lifecycle** = :func:`~jackdaw.engine.round_lifecycle.process_round_end_cards`


.. rubric:: Phase & control

::

    phase             GamePhase          Current game phase.
                                         Set by runner (initial), game (transitions).
    seed              str                Original seed string.  Set by run_init.
    seeded            bool               Whether the run is seeded.  Set by run_init.
    won               bool               True when player beats win_ante boss.
                                         Set by init (False), game (_round_won).
    round             int                Rounds completed (incremented each blind defeat).
                                         Set by init (0), game (+1 per win).
    stake             int                Stake level 1–8.  Set by run_init.
    win_ante          int                Target ante to win the run (default 8).
                                         Set by init.
    blind_on_deck     str | None         'Small' | 'Big' | 'Boss'.
                                         Set by runner (initial), game (advancement).
    blind             Blind | None       Active Blind object during a round.
                                         Set by init (None), game (_handle_select_blind).
    skips             int                Blinds skipped this run.
                                         Set by init (0), game (+1 per skip).
    selected_back_key str                Deck back key (e.g. ``'b_red'``).
                                         Set by run_init.


.. rubric:: Economy

::

    dollars            int               Current money.
                                         Set by run_init (from starting_params), mutated by
                                         game (buy/sell/score/cash-out), lifecycle (rental).
    interest_cap       int               Max money earning interest (default 25).
                                         Set by init, voucher (Seed Money→50, Money Tree→100).
    interest_amount    int               Dollars per $5 bracket (default 1).
                                         Set by init, voucher (To the Moon).
    discount_percent   int               Shop discount % (default 0).
                                         Set by init, voucher (Clearance Sale→25, Liquidation→50).
    base_reroll_cost   int               Reroll cost before per-round increases (default 5).
                                         Set by run_init.
    inflation          int               Cumulative reroll cost increase (default 0).
                                         Set by init.
    rental_rate        int               Cost per rental joker per round (default 3).
                                         Set by init.
    bankrupt_at        int               Maximum debt floor (default 0).
                                         Set by init.  Stake 5+ allows negative.
    money_per_hand     int               Dollars per hand played (Green Deck).
                                         Set by back (if applicable).
    money_per_discard  int               Dollars per discard used (Green Deck).
                                         Set by back (if applicable).
    no_interest        bool              Disable interest (Green Deck).
                                         Set by back (if applicable).


.. rubric:: Card areas

::

    hand               list[Card]        Cards currently in the player's hand.
                                         Set by game (_draw_hand, play/discard handlers).
    deck               list[Card]        Draw pile.
                                         Set by run_init (build_deck), game (shuffle/draw/return).
    discard_pile       list[Card]        Discard pile.
                                         Set by game (play/discard → extend, round_won → clear).
    jokers             list[Card]        Active joker slots.
                                         Set by game (buy/sell/pack/setting_blind mutations).
    consumables        list[Card]        Consumable card slots.
                                         Set by game (buy/use/seal effects).
    played_cards_area  list[Card]        Cards in the play area during scoring.
                                         Set by game (round_won → clear).


.. rubric:: Card area limits

::

    hand_size          int               Max hand size (default 8).
                                         Set by run_init, voucher (Paint Brush, Palette),
                                         game (The Manacle ±1), round (temp_handsize).
    joker_slots        int               Max joker slots (default 5).
                                         Set by run_init, voucher (Antimatter +1).
    consumable_slots   int               Max consumable slots (default 2).
                                         Set by run_init, voucher (Crystal Ball +1).


.. rubric:: Round resets — ``gs["round_resets"]``

Persistent per-ante values, copied into ``current_round`` each round by
:func:`~jackdaw.engine.run_init.start_round`.

::

    hands              int               Hands per round (default 4).
    discards           int               Discards per round (default 3).
    reroll_cost        int               Base reroll cost this ante (default 5).
    temp_reroll_cost   int | None        One-round reroll override (D6 Tag).
    temp_handsize      int | None        One-round hand size bonus (Juggle Tag).
    ante               int               Current ante number (starts at 1).
    blind_ante         int               Ante number for blind scaling.
    blind_states       dict[str, str]    {'Small': 'Select'|'Current'|'Skipped'|'Defeated',
                                          'Big': ..., 'Boss': ...}
    blind_choices      dict[str, str]    {'Small': 'bl_small', 'Big': 'bl_big',
                                          'Boss': <boss_key>}
    blind_tags         dict[str, str]    Tag keys awarded when skipping a blind.
                                         Set by tags (assign_ante_blinds).
    boss_rerolled      bool              Whether boss has been rerolled this ante.
    blind              Blind | None      Copy of the active Blind (also at top-level).


.. rubric:: Current round — ``gs["current_round"]``

Per-round mutable counters, reset by ``start_round``.

::

    hands_left              int          Hands remaining this round.
    hands_played            int          Hands played this round.
    discards_left           int          Discards remaining this round.
    discards_used           int          Discards used this round.
    reroll_cost             int          Current reroll cost (base + increases).
    reroll_cost_increase    int          Cumulative per-round reroll cost increase.
    free_rerolls            int          Free rerolls available (Chaos the Clown).
    jokers_purchased        int          Jokers bought from shop this round.
    dollars                 int          Dollars earned during this round's scoring.
    round_dollars           int          Running total for round earnings display.
    used_packs              list[str]    Pack keys opened this round.
    cards_flipped           int          Cards flipped face-down (boss blinds).
    voucher                 Card | None  Voucher available for purchase this ante.
    most_played_poker_hand  str          Most-played poker hand type (default 'High Card').

    current_hand            dict         Last-evaluated hand during scoring:
        chips               int              Base chips.
        mult                int              Base mult.
        handname            str              Hand type name.
        hand_level          str              Hand level string.

    Targeting cards (reset by ``reset_round_targets`` each ante):
    idol_card               dict         {'suit': str, 'rank': str}  — The Idol.
    mail_card               dict         {'rank': str}               — Mail-In Rebate.
    ancient_card            dict         {'suit': str}               — Ancient Joker.
    castle_card             dict         {'suit': str}               — Castle.


.. rubric:: Starting params — ``gs["starting_params"]``

Immutable-after-init baseline values from ``get_starting_params()``.
Mutated only by stakes and back during run_init.

::

    dollars              int             Starting money (default 4).
    hand_size            int             Starting hand size (default 8).
    discards             int             Starting discards per round (default 3).
    hands                int             Starting hands per round (default 4).
    reroll_cost          int             Starting reroll cost (default 5).
    joker_slots          int             Starting joker slots (default 5).
    consumable_slots     int             Starting consumable slots (default 2).
    ante_scaling         float           Ante chip scaling multiplier (default 1).
    no_faces             bool            Remove face cards from deck (Abandoned Deck).
    erratic_suits_and_ranks  bool        Randomize suits and ranks (Erratic Deck).


.. rubric:: Round bonus — ``gs["round_bonus"]``

One-shot bonuses consumed at the start of the next round.

::

    next_hands           int             Extra hands next round (tags).
    discards             int             Extra discards next round (tags).


.. rubric:: Shop state

::

    shop               dict              Shop config sub-dict.
        joker_max      int                   Max joker cards per shop (default 2).
                                             Increased by Overstock vouchers.
    shop_cards         list[Card]        Joker/consumable cards for sale.
                                         Set by game (_populate_shop, _reroll_shop_cards).
    shop_vouchers      list[Card]        Voucher(s) for sale.  Set by game.
    shop_boosters      list[Card]        Booster packs for sale.  Set by game.
    shop_return_phase  GamePhase         Phase to return to after pack_opening.
                                         Set by game (_handle_open_booster).


.. rubric:: Pack state

::

    pack_cards              list[Card]   Cards in the opened booster pack.
                                         Set by game (_handle_open_booster).
    pack_choices_remaining  int          Picks left in the current pack.
    pack_type               str          Pack category ('Arcana'|'Spectral'|'Celestial'|
                                         'Standard'|'Buffoon').
    pack_hand               list[Card]   Hand dealt for targeting during Arcana/Spectral packs.


.. rubric:: Scoring

::

    chips              int               Accumulated chips this round (toward blind.chips).
                                         Set by init (0), game (+= score_hand result).
    last_score_result  ScoreResult       Full result from the most recent score_hand call.
                                         Set by game (_handle_play_hand).
    round_earnings     RoundEarnings     Earnings breakdown for cash-out screen.
                                         Set by game (_round_won).


.. rubric:: Pool / shop rates

Weights for shop card type selection.  Modified by vouchers.

::

    edition_rate       float             Edition spawn weight (default 1).  Hone→2, Glow Up→4.
    joker_rate         float             Joker type weight (default 20).
    tarot_rate         float             Tarot type weight (default 4).
    planet_rate        float             Planet type weight (default 4).
    spectral_rate      float             Spectral type weight (default 0).  Ghost Deck→4.
    playing_card_rate  float             Playing card type weight (default 0).
                                         Magic Trick/Illusion→4.


.. rubric:: Probabilities

::

    probabilities      dict              Probability modifiers.
        normal         int                   Base probability multiplier (default 1).
                                             Oops! All 6s sets to 2.


.. rubric:: Meta joker flags

Flags checked by :func:`~jackdaw.engine.hand_eval.evaluate_hand` when
the corresponding joker is active.

::

    four_fingers       int               Four Fingers: flushes/straights need only 4 cards.
    shortcut           int               Shortcut: straights can gap by 1 rank.
    smeared            int               Smeared Joker: hearts=diamonds, spades=clubs for flushes.
    splash             int               Splash: every card counts for scoring.

These are typically stored as counts (0 = inactive, ≥1 = active) and
are set/incremented by joker ability application.


.. rubric:: Voucher state

::

    used_vouchers           dict[str, bool]   Redeemed vouchers {key: True}.
                                              Set by run_init (starting vouchers), game (redeem).
    omen_globe              bool              Omen Globe active (spectral in standard packs).
                                              Set by voucher.
    boss_blind_rerolls      int               Boss blind rerolls remaining (-1 = unlimited).
                                              Set by voucher (Director's Cut, Retcon).
    boss_blind_reroll_cost  int               Cost per boss reroll.
                                              Set by voucher.


.. rubric:: Run tracking / statistics

::

    round_scores       dict              Aggregate run statistics:
        furthest_ante      int               Highest ante reached.
        furthest_round     int               Highest round reached.
        hand               int               Best hand score.
        poker_hand         str               Best hand type name.
        new_collection     int               New cards discovered.
        cards_played       int               Total cards played.
        cards_discarded    int               Total cards discarded.
        times_rerolled     int               Total shop rerolls.
        cards_purchased    int               Total cards purchased.
    hands_played       int               Total hands played across all rounds.
                                         Set by init (0), game (+1 per play).
    unused_discards    int               Cumulative unused discards across all rounds
                                         (Garbage Tag).  Set by game (_round_won +=).
    actions_taken      int               Total step() calls.  Set by runner.
    previous_round     dict              Snapshot: {'dollars': int} from end of last round.


.. rubric:: Joker / consumable tracking

::

    used_jokers        dict[str, bool]   Joker keys that have appeared {key: True}.
                                         For "already seen" pool checks.
    joker_usage        dict              Per-joker usage stats.  Set by init ({}).
    consumeable_usage  dict              Per-consumable type usage.  Set by init ({}).
    hand_usage         dict              Per-hand-type play counts.  Set by init ({}).
    last_tarot_planet  str | None        Center key of last Tarot/Planet used (for The Fool).
                                         Set by init (None), game (_apply_consumable_result).


.. rubric:: Deck metadata

::

    starting_deck_size     int           Cards in deck at run start (default 52).
                                         Set by run_init (after build_deck).
    starting_consumables   list[str]     Consumable keys granted by back (Magic/Ghost Deck).
                                         Set by run_init.
    sort                   str           Default hand sort direction ('desc').
                                         Set by init.


.. rubric:: Boss / blind bookkeeping

::

    bosses_used            dict[str, int]    {boss_key: times_appeared}.
                                             Set by init, tags (assign_ante_blinds).
    current_boss_streak    int               Consecutive boss defeats without loss.
                                             Set by init (0).
    tags                   dict              Tag tracking (per-ante awarded tags).
                                             Set by init ({}), game (skip blind awards).
    tag_tally              int               Number of tags awarded.  Set by init (0).
    awarded_tags           list[dict]        Tags awarded this run (appended by skip_blind).


.. rubric:: Modifiers — ``gs["modifiers"]``

Run-wide rule flags, set by stakes and challenges.

::

    no_blind_reward        dict[str, bool]   Blinds that give no payout {'Small': True}.
    scaling                int               Ante chip scaling level (1/2/3).
    enable_eternals_in_shop    bool          Eternal jokers appear (stake ≥ 4).
    enable_perishables_in_shop bool          Perishable jokers appear (stake ≥ 7).
    enable_rentals_in_shop     bool          Rental jokers appear (stake ≥ 8).
    no_extra_hand_money    bool              Unused hands give no bonus (challenge).
    money_per_hand         int               Override $/hand for unused-hand bonus.
    money_per_discard      int               Override $/discard for unused-discard bonus.
    no_interest            bool              Disable interest entirely (challenge).
    discard_cost           int               Cost per discard action (Golden Needle).


.. rubric:: Challenge state

::

    challenge          str | None        Challenge ID (e.g. ``'c_omelette_1'``).
                                         Set by run_init (if challenge provided).
    banned_keys        dict[str, bool]   Cards/vouchers banned by challenge.
                                         Set by init ({}), challenge.


.. rubric:: Pool flags

::

    pool_flags         dict              Misc flags for pool/shop logic.  Set by init ({}).
    pack_size          int               Default booster pack size (default 2).  Set by init.
    consumeable_buffer int               Consumable buffer slots (default 0).  Set by init.
    joker_buffer       int               Joker buffer slots (default 0).  Set by init.
    ecto_minus         int               Ghost deck modifier (default 1).  Set by init.
    STOP_USE           int               Internal consumable-use guard (default 0).  Set by init.
    perishable_rounds  int               Rounds until perishable debuff (default 5).  Set by init.


.. rubric:: RNG & hand levels

::

    rng                PseudoRandom      Seeded RNG instance.  Set by run_init.
    pseudorandom       dict              Raw seed state (Lua compat).  Set by init ({}).
    hand_levels        HandLevels        Hand level progression tracker.  Set by run_init.


.. rubric:: Profile (discovery tracking)

::

    profile_unlocked   set               Unlocked profile items.  Set externally.
    discovered         set               Discovered cards/items.  Set externally.
"""

from __future__ import annotations

from typing import Any


def describe_state(gs: dict[str, Any]) -> str:
    """One-line summary of game state for debugging.

    Example::

        "SELECTING_HAND ante=2 $12 hand=8 jokers=3 chips=0/450"
    """
    phase = gs.get("phase", "?")
    ante = gs.get("round_resets", {}).get("ante", "?")
    dollars = gs.get("dollars", 0)
    hand_count = len(gs.get("hand", []))
    joker_count = len(gs.get("jokers", []))
    chips = gs.get("chips", 0)
    blind = gs.get("blind")
    blind_target = getattr(blind, "chips", "?") if blind else "?"
    return (
        f"{phase} ante={ante} ${dollars} hand={hand_count} "
        f"jokers={joker_count} chips={chips}/{blind_target}"
    )
