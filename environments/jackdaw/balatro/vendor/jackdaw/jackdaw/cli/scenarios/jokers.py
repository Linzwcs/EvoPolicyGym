"""Joker validation scenarios — one per joker, all 150.

Each scenario adds a joker to both sim and live, injects specific cards that
trigger the joker's effect, plays the hand, and compares state.  Jokers that
need special setup (discard, multi-hand, state changes) use tailored flows.

Authoritative source: balatro_source/game.lua (P_CENTERS) and
balatro_source/card.lua (Card:calculate_joker).
"""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, register
from jackdaw.cli.scenarios.helpers import (
    Handle,
    add_both,
    compare_state,
    discard,
    get_hand_count,
    play_hand,
    run_joker_with_setup,
    select_blind,
    set_both,
    start_both,
)

# ---------------------------------------------------------------------------
# Bulk registration: (key, description, hand_preset_or_None)
#
# hand_preset=None → standard play [0..4] (passive / always-trigger jokers)
# hand_preset=str  → inject cards from HAND_PRESETS and play them
# ---------------------------------------------------------------------------

_JOKER_CONFIGS: list[tuple[str, str, str | None]] = [
    # -- Always trigger / passive (no specific hand needed) --
    ("j_joker", "+4 Mult", None),
    ("j_misprint", "+0-23 Mult (random)", None),
    ("j_banner", "+30 Chips per discard remaining", None),
    ("j_abstract", "+3 Mult per joker owned", None),
    ("j_supernova", "+1 Mult per times hand type played this run", None),
    ("j_raised_fist", "+2x lowest held card rank as Mult", None),
    ("j_fortune_teller", "+1 Mult per Tarot used this run", None),
    ("j_bootstraps", "+2 Mult per $5 held", None),
    ("j_blue_joker", "+2 Chips per remaining deck card", None),
    ("j_stencil", "x1 Mult per empty joker slot", None),
    ("j_stone", "+25 Chips per Stone Card in full deck", None),
    ("j_bull", "+2 Chips per dollar held", None),
    ("j_erosion", "+4 Mult per card below starting deck size", None),
    ("j_baseball", "x1.5 Mult per uncommon joker", None),
    ("j_steel_joker", "x0.2 Mult per Steel Card in full deck", None),
    ("j_drivers_license", "x3 Mult if 16+ enhanced cards in deck", None),
    ("j_loyalty_card", "x4 Mult every 5 hands played", None),
    ("j_obelisk", "x0.2 Mult per consecutive non-most-played hand", None),
    ("j_ice_cream", "100 Chips, -5 per hand played", None),
    ("j_ride_the_bus", "+1 Mult per consecutive non-face-card hand", None),
    ("j_flash", "+2 Mult per reroll in shop (permanent)", None),
    ("j_popcorn", "+20 Mult, -4 per round played", None),
    ("j_red_card", "+3 Mult per skip (permanent)", None),
    ("j_campfire", "x0.25 per card sold, resets on boss blind defeat", None),
    ("j_constellation", "x0.1 per Planet used (permanent)", None),
    ("j_hologram", "x0.25 per card added to deck (permanent)", None),
    ("j_lucky_cat", "x0.25 per Lucky card triggered (permanent)", None),
    ("j_glass", "x0.75 per Glass Card destroyed (permanent)", None),
    ("j_swashbuckler", "Mult = total sell value of owned jokers", None),
    ("j_throwback", "x0.25 Mult per blind skipped this run", None),
    ("j_caino", "x1 Mult, +x1 per face card destroyed", None),
    # Economy (passive / end-of-round)
    ("j_golden", "+$4 at end of round", None),
    ("j_to_the_moon", "+$1 interest per $5 held", None),
    ("j_cloud_9", "+$1 per 9 in full deck at end of round", None),
    ("j_satellite", "+$1 per unique Planet used at end of round", None),
    ("j_delayed_grat", "+$2 per discard remaining at end of round", None),
    ("j_rocket", "+$1 at end of round, +$2 per boss blind defeated", None),
    ("j_egg", "+$3 sell value per round", None),
    ("j_gift", "+$1 to sell value of jokers and consumables at end of round", None),
    ("j_credit_card", "Allows going $20 into debt", None),
    # Passive / non-scoring
    ("j_chaos", "+1 free reroll per shop", None),
    ("j_juggler", "+1 hand size", None),
    ("j_drunkard", "+1 discard per round", None),
    ("j_four_fingers", "Flushes/straights need only 4 cards", None),
    ("j_shortcut", "Straights can gap by 1 rank", None),
    ("j_pareidolia", "All cards count as face cards", None),
    ("j_splash", "All cards score", None),
    ("j_smeared", "Hearts=Diamonds, Spades=Clubs for suits", None),
    ("j_oops", "Doubles all probability rolls", None),
    ("j_stuntman", "+250 Chips, -2 hand size", None),
    ("j_merry_andy", "+3 discards, -1 hand size", None),
    ("j_troubadour", "+2 hand size, -1 hand per round", None),
    ("j_turtle_bean", "+5 hand size, -1 per round", None),
    ("j_burglar", "+3 hands, lose all discards when blind selected", None),
    ("j_invisible", "After 2 rounds, sell to duplicate random joker", None),
    ("j_diet_cola", "Sell for free Double Tag", None),
    ("j_blueprint", "Copies joker to the right", None),
    ("j_brainstorm", "Copies leftmost joker", None),
    ("j_perkeo", "Duplicates 1 consumable with Negative at end of shop", None),
    ("j_astronomer", "Planet cards in shop are free", None),
    ("j_ring_master", "Showman: cards can repeat in pool", None),
    ("j_mr_bones", "Prevents death if chips >= 25% of blind", None),
    ("j_chicot", "Disables boss blind effect", None),
    ("j_matador", "+$8 on boss blind trigger", None),
    ("j_madness", "x0.5 Mult on blind select, destroys random joker", None),
    ("j_ceremonial", "+Mult from sell value when joker sold to right", None),
    ("j_hallucination", "1 in 2 chance of Tarot when opening booster pack", None),
    ("j_gros_michel", "+15 Mult, 1 in 6 chance to self-destruct", None),
    ("j_cavendish", "x3 Mult, 1 in 1000 chance to self-destruct", None),
    # Card creation on blind select (tested by the select_blind call)
    ("j_marble", "Adds Stone Card to deck when blind selected", None),
    ("j_riff_raff", "Creates 2 Common Jokers when blind selected", None),
    ("j_cartomancer", "Creates Tarot when blind selected", None),
    ("j_certificate", "Gives random playing card with random seal at round start", None),
    # -- Pair triggers --
    ("j_jolly", "+8 Mult if hand contains Pair", "PAIR"),
    ("j_sly", "+50 Chips if hand contains Pair", "PAIR"),
    ("j_duo", "x2 Mult if hand contains Pair", "PAIR"),
    ("j_hanging_chad", "Retrigger first card scored 2 times", "PAIR"),
    ("j_selzer", "Retrigger all played cards (10 uses)", "PAIR"),
    ("j_dna", "Copies first played card if first hand of round", "PAIR"),
    ("j_space", "1 in 4 chance to level up played hand type", "PAIR"),
    ("j_hiker", "+5 permanent Chips per card scored", "PAIR"),
    # -- Three of a Kind triggers --
    ("j_zany", "+12 Mult if hand contains Three of a Kind", "THREE_KIND"),
    ("j_wily", "+100 Chips if hand contains Three of a Kind", "THREE_KIND"),
    ("j_trio", "x3 Mult if hand contains Three of a Kind", "THREE_KIND"),
    # -- Two Pair triggers --
    ("j_mad", "+10 Mult if hand contains Two Pair", "TWO_PAIR"),
    ("j_clever", "+80 Chips if hand contains Two Pair", "TWO_PAIR"),
    ("j_trousers", "+2 Mult if hand contains Two Pair (permanent)", "TWO_PAIR"),
    # -- Four of a Kind --
    ("j_family", "x4 Mult if hand contains Four of a Kind", "FOUR_KIND"),
    # -- Straight triggers --
    ("j_crazy", "+12 Mult if hand contains Straight", "STRAIGHT"),
    ("j_devious", "+100 Chips if hand contains Straight", "STRAIGHT"),
    ("j_order", "x3 Mult if hand contains Straight", "STRAIGHT"),
    ("j_runner", "+15 Chips if hand contains Straight (permanent)", "STRAIGHT"),
    # -- Flush triggers (use Hearts) --
    ("j_droll", "+10 Mult if hand contains Flush", "FLUSH_HEARTS"),
    ("j_crafty", "+80 Chips if hand contains Flush", "FLUSH_HEARTS"),
    ("j_tribe", "x2 Mult if hand contains Flush", "FLUSH_HEARTS"),
    # -- Suit-per-card: Hearts --
    ("j_lusty_joker", "+3 Mult per Heart card scored", "FLUSH_HEARTS"),
    ("j_bloodstone", "1 in 2 chance x1.5 Mult per Heart scored", "FLUSH_HEARTS"),
    # -- Suit-per-card: Diamonds --
    ("j_greedy_joker", "+3 Mult per Diamond card scored", "FLUSH_DIAMONDS"),
    ("j_rough_gem", "+$1 per Diamond scored", "FLUSH_DIAMONDS"),
    # -- Suit-per-card: Spades --
    ("j_wrathful_joker", "+3 Mult per Spade card scored", "FLUSH_SPADES"),
    ("j_arrowhead", "+50 Chips per Spade scored", "FLUSH_SPADES"),
    # -- Suit-per-card: Clubs --
    ("j_gluttenous_joker", "+3 Mult per Club card scored", "FLUSH_CLUBS"),
    ("j_onyx_agate", "+7 Mult per Club scored", "FLUSH_CLUBS"),
    # -- All suits present --
    ("j_flower_pot", "x3 Mult if hand has Diamond+Club+Heart+Spade", "ALL_SUITS"),
    ("j_seeing_double", "x2 Mult if hand has Club and suit matching card", "ALL_SUITS"),
    ("j_ancient", "x1.5 Mult per card of specific suit scored", "ALL_SUITS"),
    # -- Face card triggers --
    ("j_scary_face", "+30 Chips per face card scored", "FACE_CARDS"),
    ("j_smiley", "+5 Mult per face card scored", "FACE_CARDS"),
    ("j_photograph", "x2 Mult for first face card scored", "FACE_CARDS"),
    ("j_sock_and_buskin", "Retrigger face cards", "FACE_CARDS"),
    ("j_business", "1 in 2 chance of $2 per face card scored", "FACE_CARDS"),
    ("j_reserved_parking", "1 in 2 chance of $1 per face card held", "FACE_CARDS"),
    ("j_midas_mask", "Face cards become Gold when played", "FACE_CARDS"),
    # -- Kings / Queens --
    ("j_triboulet", "x2 Mult per King or Queen scored", "KINGS_QUEENS"),
    ("j_baron", "x1.5 Mult per King held in hand", "KINGS_QUEENS"),
    # -- Rank-conditional --
    ("j_even_steven", "+4 Mult per even rank scored", "EVEN_RANKS"),
    ("j_odd_todd", "+31 Chips per odd rank scored", "ODD_RANKS"),
    ("j_fibonacci", "+8 Mult per Ace/2/3/5/8 scored", "FIBONACCI"),
    ("j_scholar", "+20 Chips and +4 Mult per Ace scored", "WITH_ACES"),
    ("j_walkie_talkie", "+10 Chips and +4 Mult per 10 or 4 scored", "WITH_TENS_FOURS"),
    ("j_hack", "Retrigger 2/3/4/5 cards", "HACK_RANKS"),
    ("j_wee", "+8 Chips per 2 scored (permanent)", "WITH_WEE_TWOS"),
    ("j_8_ball", "1 in 4 chance to create Tarot per 8 scored", "WITH_EIGHTS"),
    # -- Special hand combos --
    ("j_superposition", "Creates Tarot if hand has Ace and Straight", "STRAIGHT_ACE"),
    ("j_seance", "Creates Spectral if hand is Straight Flush", "STRAIGHT_FLUSH"),
    ("j_blackboard", "x3 Mult if all held cards are Spades or Clubs", "SPADES_CLUBS"),
    # -- ID/random target (HIGH_CARD is fine, trigger depends on seed) --
    ("j_idol", "x2 Mult for specific suit+rank card", "HIGH_CARD"),
    ("j_ticket", "+$4 per Gold card scored", "HIGH_CARD"),
    ("j_mime", "Retrigger held-in-hand effects", "HIGH_CARD"),
    # -- Discard-scaling (triggers on discard, standard hand ok for play) --
    ("j_green_joker", "+1 Mult per hand, -1 per discard", None),
    ("j_ramen", "x2 Mult, -0.01 per card discarded", None),
    ("j_card_sharp", "x3 Mult if same hand type played this round already", None),
    ("j_vampire", "x0.1 per enhanced card played, removes enhancement", None),
]

for _key, _desc, _preset in _JOKER_CONFIGS:

    def _make_fn(key: str = _key, preset: str | None = _preset):  # noqa: B023
        def fn(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
            return run_joker_with_setup(sim, live, joker_key=key, hand_preset=preset, delay=delay)

        return fn

    register(
        name=f"joker_{_key[2:]}",  # strip j_ prefix
        category="jokers",
        description=_desc,
    )(_make_fn(_key, _preset))


# ---------------------------------------------------------------------------
# Special jokers that need non-standard flows
# ---------------------------------------------------------------------------


@register(
    name="joker_half",
    category="jokers",
    description="+20 Mult if hand has 3 or fewer cards",
)
def _joker_half(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim, live, joker_key="j_half", hand_preset="THREE_CARDS", play_count=3, delay=delay
    )


@register(
    name="joker_square",
    category="jokers",
    description="+4 Chips if hand has exactly 4 cards (permanent)",
)
def _joker_square(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim, live, joker_key="j_square", hand_preset="FOUR_CARDS", play_count=4, delay=delay
    )


@register(
    name="joker_acrobat",
    category="jokers",
    description="x3 Mult on final hand of round",
)
def _joker_acrobat(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_ACROBAT", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_acrobat")
    set_both(sim, live, hands=1)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="acrobat final hand")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Acrobat: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_dusk",
    category="jokers",
    description="Retrigger all played cards on final hand",
)
def _joker_dusk(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_DUSK", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_dusk")
    set_both(sim, live, hands=1)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="dusk final hand")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Dusk: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_mystic_summit",
    category="jokers",
    description="+15 Mult when 0 discards left",
)
def _joker_mystic_summit(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_MYSTIC_SUMMIT", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_mystic_summit")
    set_both(sim, live, discards=0)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="mystic_summit 0 discards")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Mystic Summit: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_shoot_the_moon",
    category="jokers",
    description="+13 Mult per Queen held in hand",
)
def _joker_shoot_the_moon(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    """Inject Queens but play non-Queens so Queens remain held."""
    start_both(sim, live, seed="J_SHOOT_THE_MOON", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_shoot_the_moon")
    # Add 2 Queens (held) + 5 non-Queens (played)
    for key in ["H_Q", "D_Q", "S_2", "C_3", "H_4", "D_5", "S_6"]:
        add_both(sim, live, key=key)
    # Play the last 5 (non-Queens), keep Queens held
    count = get_hand_count(sim)
    play_indices = list(range(count - 5, count))
    play_hand(sim, live, play_indices, delay=delay)
    diffs = compare_state(sim, live, label="shoot_the_moon queens held")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Shoot the Moon: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_sixth_sense",
    category="jokers",
    description="Destroys first played 6 for Spectral (first hand only)",
)
def _joker_sixth_sense(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim, live, joker_key="j_sixth_sense", hand_preset="WITH_SIXES", delay=delay
    )


@register(
    name="joker_vagabond",
    category="jokers",
    description="Creates Tarot if $4 or less after play",
)
def _joker_vagabond(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_VAGABOND", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_vagabond")
    set_both(sim, live, money=4)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="vagabond low money")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Vagabond: {'PASS' if not diffs else 'FAIL'}",
    )


# -- Discard-based jokers --


@register(
    name="joker_faceless",
    category="jokers",
    description="$5 if 3+ face cards discarded",
)
def _joker_faceless(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim,
        live,
        joker_key="j_faceless",
        hand_preset="HIGH_CARD",
        pre_discard=["H_J", "D_Q", "S_K"],
        delay=delay,
    )


@register(
    name="joker_mail",
    category="jokers",
    description="$5 per discarded card of target rank",
)
def _joker_mail(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_MAIL", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_mail")
    discard(sim, live, [0, 1, 2], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="mail after discard+play")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Mail: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_trading",
    category="jokers",
    description="$3 if first discard has only 1 card",
)
def _joker_trading(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_TRADING", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_trading")
    discard(sim, live, [0], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="trading after discard+play")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Trading: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_hit_the_road",
    category="jokers",
    description="x0.5 Mult per Jack discarded this round",
)
def _joker_hit_the_road(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim,
        live,
        joker_key="j_hit_the_road",
        hand_preset="HIGH_CARD",
        pre_discard=["H_J", "D_J", "S_J", "C_J"],
        delay=delay,
    )


@register(
    name="joker_castle",
    category="jokers",
    description="+3 Chips per card discarded of matching suit (permanent)",
)
def _joker_castle(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim,
        live,
        joker_key="j_castle",
        hand_preset="HIGH_CARD",
        pre_discard=["H_2", "H_3", "H_4"],
        delay=delay,
    )


@register(
    name="joker_burnt",
    category="jokers",
    description="Upgrade discarded hand type level",
)
def _joker_burnt(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    return run_joker_with_setup(
        sim,
        live,
        joker_key="j_burnt",
        hand_preset="PAIR",
        pre_discard=["H_A", "D_A"],  # discard a pair to level up Pair
        delay=delay,
    )


@register(
    name="joker_yorick",
    category="jokers",
    description="x1 Mult, needs 23 discards then x5",
)
def _joker_yorick(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    # Yorick needs many discards; just test basic discard tracking
    start_both(sim, live, seed="J_YORICK", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_yorick")
    discard(sim, live, [0, 1, 2], delay=delay)
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="yorick after discard+play")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Yorick: {'PASS' if not diffs else 'FAIL'}",
    )


@register(
    name="joker_todo_list",
    category="jokers",
    description="$4 if poker hand matches target",
)
def _joker_todo_list(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    start_both(sim, live, seed="J_TODO", delay=delay)
    select_blind(sim, live, delay=delay)
    add_both(sim, live, key="j_todo_list")
    play_hand(sim, live, [0, 1, 2, 3, 4], delay=delay)
    diffs = compare_state(sim, live, label="todo_list after play")
    return ScenarioResult(
        passed=not diffs,
        diffs=diffs,
        details=f"Todo List: {'PASS' if not diffs else 'FAIL'}",
    )


@register(name="joker_luchador", category="jokers", description="Sell to disable boss blind")
def _joker_luchador(sim: Handle, live: Handle, *, delay: float = 0.3) -> ScenarioResult:
    # Luchador requires selling during a round (SELECTING_HAND), but the
    # balatrobot API only supports sell in SHOP state.  Skip this scenario.
    return ScenarioResult(
        passed=True,
        details="Luchador: SKIP (balatrobot sell API requires SHOP state)",
    )
