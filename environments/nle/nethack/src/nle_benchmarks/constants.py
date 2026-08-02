"""Version-pinned public constants for the NLE score profile."""

from evopolicygym.policy import PolicyValue

UPSTREAM_VERSION = "1.3.0"
NETHACK_VERSION = "3.6.7"
BENCHMARK_ID = "nle/NetHackScore-v0/mean-return-v1"
CHARACTER = "mon-hum-neu-mal"
PENALTY_MODE = "constant"
PENALTY_STEP = -0.01
PENALTY_TIME = 0.0
FEEDBACK_SCOPE_KEY = "feedback_scope"
PUBLIC_FEEDBACK_SCOPE = "public_training"
AGGREGATE_FEEDBACK_SCOPE = "aggregate_only"

OBSERVATION_KEYS = (
    "glyphs",
    "chars",
    "colors",
    "blstats",
    "message",
    "inv_glyphs",
    "inv_strs",
    "inv_letters",
    "inv_oclasses",
    "misc",
)

NETHACK_OPTIONS = (
    "autopickup",
    "color",
    "disclose:+i +a +v +g +c +o",
    "mention_walls",
    "nobones",
    "nocmdassist",
    "nolegacy",
    "nosparkle",
    "pickup_burden:unencumbered",
    "pickup_types:$?!/",
    "runmode:teleport",
    "showexp",
    "showscore",
    "time",
)

RAW_ACTIONS = (
    13,
    107,
    108,
    106,
    104,
    117,
    110,
    98,
    121,
    75,
    76,
    74,
    72,
    85,
    78,
    66,
    89,
    60,
    62,
    46,
    4,
    101,
    115,
)

ACTION_MEANINGS = (
    "more",
    "north",
    "east",
    "south",
    "west",
    "northeast",
    "southeast",
    "southwest",
    "northwest",
    "run_north",
    "run_east",
    "run_south",
    "run_west",
    "run_northeast",
    "run_southeast",
    "run_southwest",
    "run_northwest",
    "up",
    "down",
    "wait",
    "kick",
    "eat",
    "search",
)

ACTION_SPACE: dict[str, PolicyValue] = {
    "type": "discrete",
    "values": list(range(len(ACTION_MEANINGS))),
    "meaning": {
        str(index): meaning for index, meaning in enumerate(ACTION_MEANINGS)
    },
    "raw_key_code": {
        str(index): value for index, value in enumerate(RAW_ACTIONS)
    },
}

BLSTAT_NAMES = (
    "x",
    "y",
    "strength_25",
    "strength_125",
    "dexterity",
    "constitution",
    "intelligence",
    "wisdom",
    "charisma",
    "score",
    "hit_points",
    "max_hit_points",
    "depth",
    "gold",
    "energy",
    "max_energy",
    "armor_class",
    "monster_level",
    "experience_level",
    "experience_points",
    "turn",
    "hunger",
    "encumbrance",
    "dungeon_number",
    "dungeon_level",
    "condition_mask",
    "alignment",
)

CONDITION_BITS = (
    (0x0001, "stoned"),
    (0x0002, "slimed"),
    (0x0004, "strangled"),
    (0x0008, "food_poisoned"),
    (0x0010, "terminally_ill"),
    (0x0020, "blind"),
    (0x0040, "deaf"),
    (0x0080, "stunned"),
    (0x0100, "confused"),
    (0x0200, "hallucinating"),
    (0x0400, "levitating"),
    (0x0800, "flying"),
    (0x1000, "riding"),
)

__all__ = [
    "ACTION_MEANINGS",
    "ACTION_SPACE",
    "BENCHMARK_ID",
    "BLSTAT_NAMES",
    "CHARACTER",
    "CONDITION_BITS",
    "AGGREGATE_FEEDBACK_SCOPE",
    "FEEDBACK_SCOPE_KEY",
    "NETHACK_OPTIONS",
    "NETHACK_VERSION",
    "OBSERVATION_KEYS",
    "PENALTY_MODE",
    "PENALTY_STEP",
    "PENALTY_TIME",
    "PUBLIC_FEEDBACK_SCOPE",
    "RAW_ACTIONS",
    "UPSTREAM_VERSION",
]
