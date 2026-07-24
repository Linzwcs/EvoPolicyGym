"""Tests for the Balatro-compatible PRNG system.

Test strategy:
  - Validate pseudohash against known Lua outputs for specific strings
  - Validate PseudoRandom.seed() stream advancement via Lua ground truth
  - Validate PseudoRandom class API: random, element, shuffle, predict_seed
  - Validate functional wrappers (pseudoseed, pseudorandom, etc.)

Ground-truth values were obtained by running the original Lua functions
(misc_functions.lua:279-320) via lupa, matching to 15 decimal places.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from jackdaw.engine.rng import (
    PseudoRandom,
    pseudohash,
    pseudoseed,
)

# Tolerance: Lua uses IEEE 754 doubles, same as Python. pseudohash has no
# truncation so we require near-exact equality. pseudoseed truncates to 13
# decimal places via string.format("%.13f", ...).
HASH_TOL = 1e-15
SEED_TOL = 1e-13


# ============================================================================
# Ground-truth from Lua (via lupa)
# ============================================================================

PSEUDOHASH_TRUTH = {
    "TESTSEED": 0.319272078222355,
    "A3K9NZ2B": 0.946384286395556,
    "a": 0.641368674218143,
    "bossTUTORIAL": 0.174561306470309,
}

# pseudoseed ground truth: {seed_str: {stream_key: [(result, stored), ...]}}
PSEUDOSEED_TRUTH = {
    "TESTSEED": {
        "boss": [
            (0.655250828936977, 0.991229579651600),
            (0.581457451664827, 0.843642825107300),
            (0.454214620624177, 0.589157163026000),
            (0.234808236884227, 0.150344395546100),
            (0.356483101926677, 0.393694125631000),
        ],
    },
}

# predict_seed ground truth
PREDICT_SEED_TRUTH = [
    ("Joker4", "TESTSEED", 0.236212407946077),
    ("Joker4", "TUTORIAL", 0.585922460919508),
    ("boss", "A3K9NZ2B", 0.894786055642878),
]


# ============================================================================
# pseudohash tests
# ============================================================================


class TestPseudohash:
    """pseudohash: string -> float."""

    @pytest.mark.parametrize("s,expected", PSEUDOHASH_TRUTH.items())
    def test_matches_lua(self, s: str, expected: float):
        assert abs(pseudohash(s) - expected) < HASH_TOL, (
            f"pseudohash({s!r}): {pseudohash(s):.15f} != {expected:.15f}"
        )


# ============================================================================
# PseudoRandom class — minimal smoke tests (oracle sections provide coverage)
# ============================================================================


class TestPseudoRandomSeed:
    """PseudoRandom.seed(): stream independence."""

    def test_streams_are_independent(self):
        """Advancing one stream must not affect another."""
        prng = PseudoRandom("TESTSEED")
        prng.seed("alpha")
        prng.seed("beta")
        alpha_val = prng.state["alpha"]
        for _ in range(10):
            prng.seed("beta")
        assert prng.state["alpha"] == alpha_val


class TestFunctionalPseudoseed:
    def test_matches_class_api(self):
        prng = PseudoRandom("TESTSEED")
        class_results = [prng.seed("boss") for _ in range(5)]

        state = {"seed": "TESTSEED", "hashed_seed": pseudohash("TESTSEED")}
        func_results = [pseudoseed("boss", state) for _ in range(5)]

        for i, (c, f) in enumerate(zip(class_results, func_results)):
            assert abs(c - f) < SEED_TOL, f"Call {i}: class={c:.15f} func={f:.15f}"


# ============================================================================
# RNG sequence integration — simulates real Balatro run
# ============================================================================

FLOAT_TOL = 1e-14

# Pools (simplified versions of the real game pools, matching the Lua oracle)
BOSS_POOL = {
    "bl_hook": "bl_hook",
    "bl_ox": "bl_ox",
    "bl_wall": "bl_wall",
    "bl_wheel": "bl_wheel",
    "bl_arm": "bl_arm",
    "bl_club": "bl_club",
    "bl_fish": "bl_fish",
    "bl_psychic": "bl_psychic",
    "bl_goad": "bl_goad",
    "bl_water": "bl_water",
}

TAG_POOL = {
    "tag_uncommon": "tag_uncommon",
    "tag_rare": "tag_rare",
    "tag_negative": "tag_negative",
    "tag_foil": "tag_foil",
    "tag_holo": "tag_holo",
    "tag_polychrome": "tag_polychrome",
    "tag_investment": "tag_investment",
    "tag_voucher": "tag_voucher",
    "tag_boss": "tag_boss",
    "tag_standard": "tag_standard",
    "tag_charm": "tag_charm",
    "tag_meteor": "tag_meteor",
    "tag_buffoon": "tag_buffoon",
    "tag_handy": "tag_handy",
    "tag_garbage": "tag_garbage",
    "tag_ethereal": "tag_ethereal",
    "tag_coupon": "tag_coupon",
    "tag_double": "tag_double",
    "tag_juggle": "tag_juggle",
    "tag_d6": "tag_d6",
}

JOKER_POOL = {
    "j_joker": "j_joker",
    "j_greedy_joker": "j_greedy_joker",
    "j_lusty_joker": "j_lusty_joker",
    "j_wrathful_joker": "j_wrathful_joker",
    "j_jolly": "j_jolly",
    "j_zany": "j_zany",
    "j_mad": "j_mad",
    "j_crazy": "j_crazy",
    "j_half": "j_half",
    "j_stencil": "j_stencil",
}

# LuaJIT 2.1 ground truth for seed "TESTSEED"
EXPECTED_DECK = [
    17,
    21,
    14,
    36,
    29,
    31,
    35,
    41,
    43,
    34,
    44,
    4,
    49,
    50,
    32,
    26,
    10,
    42,
    12,
    27,
    6,
    19,
    48,
    52,
    51,
    38,
    8,
    20,
    11,
    1,
    30,
    37,
    13,
    25,
    47,
    22,
    18,
    24,
    16,
    2,
    40,
    39,
    15,
    28,
    5,
    45,
    7,
    3,
    33,
    9,
    23,
    46,
]

EXPECTED_BOSS = "bl_goad"
EXPECTED_TAGS = ["tag_voucher", "tag_investment"]

EXPECTED_SHOP = [
    {"cdt": 0.323673317736492, "rarity": 0.127372906355584, "joker": "j_joker"},
    {"cdt": 0.537613412513260, "rarity": 0.777377281373973, "joker": "j_mad"},
    {"cdt": 0.134618964202504, "rarity": 0.309657059223484, "joker": "j_joker"},
]

EXPECTED_REROLLS = [
    {"cdt": 0.605297975170146, "rarity": 0.863258824107398, "joker": "j_greedy_joker"},
    {"cdt": 0.572515377396789, "rarity": 0.854217398275141, "joker": "j_lusty_joker"},
    {"cdt": 0.371821413620441, "rarity": 0.298228735213785, "joker": "j_crazy"},
]

EXPECTED_BOSS_CALL2 = 0.58145745166482732


class TestGameSequence:
    """Simulate the first few RNG calls of a real Balatro run."""

    @pytest.fixture
    def prng(self) -> PseudoRandom:
        return PseudoRandom("TESTSEED")

    def test_step1_deck_shuffle(self, prng: PseudoRandom):
        """Deck shuffle uses 'shuffle' stream, produces known order."""
        sv = prng.seed("shuffle")
        deck = list(range(1, 53))
        prng.shuffle(deck, sv)
        assert deck == EXPECTED_DECK

    def test_step2_boss_selection(self, prng: PseudoRandom):
        """Boss selection uses 'boss' stream, picks from sorted pool."""
        prng.seed("shuffle")
        sv = prng.seed("boss")
        _, boss_key = prng.element(BOSS_POOL, sv)
        assert boss_key == EXPECTED_BOSS

    def test_step3_tag_generation(self, prng: PseudoRandom):
        """Two tags from 'Tag1' stream — consecutive calls, same stream."""
        prng.seed("shuffle")
        prng.seed("boss")
        for expected_tag in EXPECTED_TAGS:
            sv = prng.seed("Tag1")
            _, tag_key = prng.element(TAG_POOL, sv)
            assert tag_key == expected_tag

    def test_step4_shop_population(self, prng: PseudoRandom):
        """3 shop slots, each advancing cdt/rarity/pool streams."""
        prng.seed("shuffle")
        prng.seed("boss")
        prng.seed("Tag1")
        prng.seed("Tag1")

        ante = 1
        for slot_idx, expected in enumerate(EXPECTED_SHOP):
            sv_cdt = prng.seed(f"cdt{ante}")
            cdt_float = prng.random(sv_cdt)

            sv_rar = prng.seed(f"rarity{ante}sho")
            rarity_float = prng.random(sv_rar)

            sv_pool = prng.seed(f"Joker1sho{ante}")
            _, joker_key = prng.element(JOKER_POOL, sv_pool)

            assert abs(cdt_float - expected["cdt"]) < FLOAT_TOL, (
                f"shop[{slot_idx + 1}] cdt: {cdt_float:.15f} != {expected['cdt']:.15f}"
            )
            assert abs(rarity_float - expected["rarity"]) < FLOAT_TOL, (
                f"shop[{slot_idx + 1}] rarity: {rarity_float:.15f} != {expected['rarity']:.15f}"
            )
            assert joker_key == expected["joker"], (
                f"shop[{slot_idx + 1}] joker: {joker_key!r} != {expected['joker']!r}"
            )

    def test_step5_rerolls(self, prng: PseudoRandom):
        """3 rerolls continue advancing the same shop streams."""
        prng.seed("shuffle")
        prng.seed("boss")
        prng.seed("Tag1")
        prng.seed("Tag1")

        ante = 1
        for _ in range(3):
            prng.seed(f"cdt{ante}")
            prng.seed(f"rarity{ante}sho")
            prng.seed(f"Joker1sho{ante}")

        for reroll_idx, expected in enumerate(EXPECTED_REROLLS):
            sv_cdt = prng.seed(f"cdt{ante}")
            cdt_float = prng.random(sv_cdt)

            sv_rar = prng.seed(f"rarity{ante}sho")
            rarity_float = prng.random(sv_rar)

            sv_pool = prng.seed(f"Joker1sho{ante}")
            _, joker_key = prng.element(JOKER_POOL, sv_pool)

            assert abs(cdt_float - expected["cdt"]) < FLOAT_TOL, (
                f"reroll[{reroll_idx + 1}] cdt: {cdt_float:.15f} != {expected['cdt']:.15f}"
            )
            assert abs(rarity_float - expected["rarity"]) < FLOAT_TOL, (
                f"reroll[{reroll_idx + 1}] rarity"
            )
            assert joker_key == expected["joker"], (
                f"reroll[{reroll_idx + 1}] joker: {joker_key!r} != {expected['joker']!r}"
            )

    def test_stream_independence(self, prng: PseudoRandom):
        """Advancing shop streams must not affect the boss stream."""
        prng.seed("shuffle")
        prng.seed("boss")
        prng.seed("Tag1")
        prng.seed("Tag1")

        ante = 1
        for _ in range(6):
            prng.seed(f"cdt{ante}")
            prng.seed(f"rarity{ante}sho")
            prng.seed(f"Joker1sho{ante}")

        sv_boss2 = prng.seed("boss")
        assert abs(sv_boss2 - EXPECTED_BOSS_CALL2) < SEED_TOL, (
            f"boss call 2: {sv_boss2:.17g} != {EXPECTED_BOSS_CALL2:.17g}"
        )

    def test_full_sequence_in_one_pass(self, prng: PseudoRandom):
        """Run all steps sequentially in a single pass, verifying each."""
        ante = 1

        # 1. Deck shuffle
        sv = prng.seed("shuffle")
        deck = list(range(1, 53))
        prng.shuffle(deck, sv)
        assert deck == EXPECTED_DECK

        # 2. Boss selection
        sv = prng.seed("boss")
        _, boss = prng.element(BOSS_POOL, sv)
        assert boss == EXPECTED_BOSS

        # 3. Tags
        for exp_tag in EXPECTED_TAGS:
            sv = prng.seed("Tag1")
            _, tag = prng.element(TAG_POOL, sv)
            assert tag == exp_tag

        # 4. Shop + 5. Rerolls (6 iterations total)
        for expected in EXPECTED_SHOP + EXPECTED_REROLLS:
            sv_cdt = prng.seed(f"cdt{ante}")
            cdt = prng.random(sv_cdt)
            assert abs(cdt - expected["cdt"]) < FLOAT_TOL

            sv_rar = prng.seed(f"rarity{ante}sho")
            rar = prng.random(sv_rar)
            assert abs(rar - expected["rarity"]) < FLOAT_TOL

            sv_pool = prng.seed(f"Joker1sho{ante}")
            _, jk = prng.element(JOKER_POOL, sv_pool)
            assert jk == expected["joker"]

        # 6. Boss call 2
        sv = prng.seed("boss")
        assert abs(sv - EXPECTED_BOSS_CALL2) < SEED_TOL


# ============================================================================
# Cross-validation: Python PseudoRandom vs Lua ground truth
# (merged from test_rng_oracle.py)
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "rng_oracle_TESTSEED.json"
ORACLE_SCRIPT = PROJECT_ROOT / "scripts" / "lua_rng_oracle.lua"

# Tolerances (same as main tests above)
ORACLE_HASH_TOL = 1e-15
ORACLE_SEED_TOL = 1e-13


def load_fixture() -> dict:
    """Load the pre-generated Lua oracle fixture."""
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def find_lua() -> str | None:
    """Find a Lua interpreter (luajit preferred, then lua, lua5.1, etc.)."""
    for name in ["luajit", "lua", "lua5.1", "lua5.4", "lua54", "lua51"]:
        path = shutil.which(name)
        if path:
            return path
    # Check common Windows install locations
    for candidate in [
        Path.home() / "AppData/Local/Programs/LuaJIT/bin/luajit.exe",
        Path("C:/Program Files/LuaJIT/bin/luajit.exe"),
        Path("C:/Program Files (x86)/LuaJIT/bin/luajit.exe"),
    ]:
        if candidate.exists():
            return str(candidate)
    return None


def run_lua_oracle(seed: str, lua_path: str) -> dict:
    """Run the Lua oracle script and parse its JSON output."""
    result = subprocess.run(
        [lua_path, str(ORACLE_SCRIPT), seed],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.fail(f"Lua oracle failed:\nstderr: {result.stderr}\nstdout: {result.stdout}")
    # The output is a wrapper with a "seeds" array
    data = json.loads(result.stdout)
    if "seeds" in data:
        return data["seeds"][0]
    return data


class TestFixtureOracle:
    """Validate Python RNG against pre-generated Lua fixture (TESTSEED)."""

    @pytest.fixture(scope="class")
    def fixture(self) -> dict:
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
        return load_fixture()

    @pytest.fixture(scope="class")
    def prng(self) -> PseudoRandom:
        return PseudoRandom("TESTSEED")

    def test_hashed_seed(self, fixture):
        expected = fixture["hashed_seed"]
        actual = pseudohash("TESTSEED")
        assert abs(actual - expected) < ORACLE_HASH_TOL, (
            f"hashed_seed: {actual:.15f} != {expected:.15f}"
        )

    def test_pseudohash_stream_keys(self, fixture):
        """Verify pseudohash(key + seed) for all stream keys."""
        for key, expected in fixture["pseudohash"].items():
            actual = pseudohash(key + "TESTSEED")
            assert abs(actual - expected) < ORACLE_HASH_TOL, (
                f"pseudohash({key}+TESTSEED): {actual:.15f} != {expected:.15f}"
            )

    @pytest.mark.parametrize(
        "stream_key",
        [
            "boss",
            "shuffle",
            "lucky_mult",
            "rarity1",
            "stdset1",
            "cdt1",
            "front1",
            "edition_generic",
        ],
    )
    def test_pseudoseed_sequence(self, fixture, stream_key):
        """10 consecutive pseudoseed advances must match Lua for each stream."""
        prng = PseudoRandom("TESTSEED")
        expected_seq = fixture["pseudoseed"][stream_key]

        for entry in expected_seq:
            call_num = entry["call"]
            expected_result = entry["result"]
            expected_stored = entry["stored"]

            result = prng.seed(stream_key)
            stored = prng.state[stream_key]

            assert abs(result - expected_result) < ORACLE_SEED_TOL, (
                f"stream={stream_key!r} call {call_num}: "
                f"result {result:.15f} != {expected_result:.15f}"
            )
            assert abs(stored - expected_stored) < ORACLE_SEED_TOL, (
                f"stream={stream_key!r} call {call_num}: "
                f"stored {stored:.15f} != {expected_stored:.15f}"
            )

    def test_predict_seed(self, fixture):
        """Stateless predict_seed must match Lua."""
        prng = PseudoRandom("TESTSEED")
        for entry in fixture["predict_seed"]:
            key = entry["key"]
            predict_with = entry["predict_with"]
            expected = entry["result"]
            actual = prng.predict_seed(key, predict_with)
            assert abs(actual - expected) < ORACLE_SEED_TOL, (
                f"predict_seed({key!r}, {predict_with!r}): {actual:.15f} != {expected:.15f}"
            )

    def test_all_streams_independent(self, fixture):
        """Advancing all 8 streams interleaved must still match fixture.

        This catches bugs where advancing one stream corrupts another.
        """
        prng = PseudoRandom("TESTSEED")
        streams = list(fixture["pseudoseed"].keys())

        # Advance all streams call-by-call (interleaved)
        for call_idx in range(10):
            for stream_key in streams:
                entry = fixture["pseudoseed"][stream_key][call_idx]
                result = prng.seed(stream_key)
                assert abs(result - entry["result"]) < ORACLE_SEED_TOL, (
                    f"interleaved: stream={stream_key} call {call_idx + 1}: "
                    f"{result:.15f} != {entry['result']:.15f}"
                )

    def test_fixture_has_expected_structure(self, fixture):
        """Sanity check: fixture has the data we expect."""
        assert fixture["seed"] == "TESTSEED"
        assert "hashed_seed" in fixture
        assert len(fixture["pseudoseed"]) == 8
        assert len(fixture["pseudoseed"]["boss"]) == 10
        assert len(fixture["predict_seed"]) == 3


class TestLiveLuaOracle:
    """Run the Lua oracle script live and validate against Python."""

    @pytest.fixture(scope="class")
    def lua_path(self):
        path = find_lua()
        if not path:
            pytest.skip(
                "No Lua interpreter found. Install lua, lua5.1, or luajit "
                "to enable live Lua cross-validation."
            )
        return path

    @pytest.fixture(scope="class")
    def oracle_data(self, lua_path) -> dict:
        """Run oracle for TESTSEED and parse output."""
        return run_lua_oracle("TESTSEED", lua_path)

    def test_hashed_seed(self, oracle_data):
        expected = oracle_data["hashed_seed"]
        actual = pseudohash("TESTSEED")
        assert abs(actual - expected) < ORACLE_HASH_TOL

    @pytest.mark.parametrize(
        "stream_key",
        [
            "boss",
            "shuffle",
            "lucky_mult",
            "rarity1",
            "stdset1",
            "cdt1",
            "front1",
            "edition_generic",
        ],
    )
    def test_pseudoseed_sequence(self, oracle_data, stream_key):
        """10 advances must match live Lua output."""
        prng = PseudoRandom("TESTSEED")
        expected_seq = oracle_data["pseudoseed"][stream_key]

        for entry in expected_seq:
            call_num = entry["call"]
            expected_result = entry["result"]
            result = prng.seed(stream_key)
            assert abs(result - expected_result) < ORACLE_SEED_TOL, (
                f"live lua: stream={stream_key!r} call {call_num}: "
                f"{result:.15f} != {expected_result:.15f}"
            )

    def test_predict_seed(self, oracle_data):
        prng = PseudoRandom("TESTSEED")
        for entry in oracle_data["predict_seed"]:
            expected = entry["result"]
            actual = prng.predict_seed(entry["key"], entry["predict_with"])
            assert abs(actual - expected) < ORACLE_SEED_TOL

    @pytest.mark.parametrize("seed_str", ["A3K9NZ2B", "TUTORIAL"])
    def test_other_seeds(self, lua_path, seed_str):
        """Cross-validate additional seeds beyond TESTSEED."""
        oracle = run_lua_oracle(seed_str, lua_path)

        # hashed_seed
        assert abs(pseudohash(seed_str) - oracle["hashed_seed"]) < ORACLE_HASH_TOL

        # All pseudoseed streams
        for stream_key, expected_seq in oracle["pseudoseed"].items():
            prng_stream = PseudoRandom(seed_str)  # fresh per stream
            for entry in expected_seq:
                result = prng_stream.seed(stream_key)
                assert abs(result - entry["result"]) < ORACLE_SEED_TOL, (
                    f"seed={seed_str!r} stream={stream_key!r} call {entry['call']}"
                )


# ============================================================================
# Full-pipeline LuaJIT validation (pseudoseed -> TW223 -> output)
# ============================================================================

PIPELINE_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "rng_pipeline_TESTSEED.json"
PIPELINE_FLOAT_TOL = 1e-14


class TestFullPipelineOracle:
    """Validate the complete pseudorandom pipeline against LuaJIT.

    This tests the end-to-end chain: PseudoRandom.seed() produces a float
    that seeds LuaJIT's TW223 PRNG, which then produces the final
    float/int output.  The fixture was generated by running the actual
    pseudoseed -> math.randomseed -> math.random sequence in LuaJIT 2.1.

    If these tests pass, the Python RNG is bit-compatible with LuaJIT.
    """

    @pytest.fixture(scope="class")
    def pipeline(self) -> dict:
        if not PIPELINE_FIXTURE.exists():
            pytest.skip(f"Pipeline fixture not found: {PIPELINE_FIXTURE}")
        with open(PIPELINE_FIXTURE) as f:
            return json.load(f)

    @pytest.mark.parametrize("stream_key", ["boss", "shuffle", "lucky_mult", "rarity1"])
    def test_pseudorandom_float(self, pipeline, stream_key):
        """PseudoRandom.random(key) float output must match LuaJIT."""
        prng = PseudoRandom("TESTSEED")
        expected_seq = pipeline["pseudorandom_pipeline"][stream_key]

        for entry in expected_seq:
            result = prng.random(stream_key)
            expected = entry["float"]
            assert abs(result - expected) < PIPELINE_FLOAT_TOL, (
                f"stream={stream_key!r} call {entry['call']}: "
                f"float {result:.17g} != {expected:.17g}"
            )

    @pytest.mark.parametrize("stream_key", ["boss", "shuffle", "lucky_mult", "rarity1"])
    def test_pseudorandom_int(self, pipeline, stream_key):
        """PseudoRandom.random(key, 1, 10) int output must match LuaJIT."""
        prng = PseudoRandom("TESTSEED")
        expected_seq = pipeline["pseudorandom_pipeline"][stream_key]

        for entry in expected_seq:
            result = prng.random(stream_key, min_val=1, max_val=10)
            expected = entry["int"]
            assert result == expected, (
                f"stream={stream_key!r} call {entry['call']}: int {result} != {expected}"
            )

    def test_pseudoshuffle(self, pipeline):
        """PseudoRandom.shuffle() must match LuaJIT pseudoshuffle."""
        prng = PseudoRandom("TESTSEED")

        for entry in pipeline["shuffle_pipeline"]:
            sv = prng.seed("shuffle")
            lst = list(range(1, 11))
            prng.shuffle(lst, sv)
            assert lst == entry["result"], (
                f"shuffle trial {entry['trial']}: {lst} != {entry['result']}"
            )

    def test_pseudorandom_element(self, pipeline):
        """PseudoRandom.element() must match LuaJIT pseudorandom_element."""
        prng = PseudoRandom("TESTSEED")
        table = {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "epsilon": 5}

        for entry in pipeline["element_pipeline"]:
            sv = prng.seed("element_test")
            val, key = prng.element(table, sv)
            assert key == entry["key"], (
                f"element trial {entry['trial']}: key {key!r} != {entry['key']!r}"
            )
            assert val == entry["value"], (
                f"element trial {entry['trial']}: val {val} != {entry['value']}"
            )


# ============================================================================
# Bugged seeds — nan collapse (e.g. erratic 7LB2WVPK)
# ============================================================================


class TestBuggedSeedNanCollapse:
    """Seeds whose pseudohash hits num == 0 mid-loop must collapse to nan.

    Lua semantics: 1.x/0 = inf, then inf % 1 = nan, and the nan survives the
    %.13f string round-trip (LuaJIT tonumber("nan") = nan). math.randomseed(nan)
    deterministically pins the TW223 stream, which is why community "bugged
    seeds" like erratic 7LB2WVPK deal 52 identical cards. Ground truth from
    LuaJIT 2.1: randomseed(nan) -> random(1, 52) == 52 on every draw, and the
    second float draw is 0.44933739017092433.
    """

    def test_pseudohash_nan(self):
        import math

        assert math.isnan(pseudohash("erratic7LB2WVPK"))

    def test_nan_stream_is_pinned(self):
        prng = PseudoRandom("7LB2WVPK")
        assert prng.random("erratic", 1, 52) == 52
        assert prng.random("erratic", 1, 52) == 52
        a = prng.random("erratic")
        b = prng.random("erratic")
        assert a == b  # stream is pinned: every reseed yields the same draw
        assert a >= 51 / 52  # consistent with random(1, 52) == 52

    def test_erratic_freak_deck(self):
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_erratic", 1, "7LB2WVPK")
        identities = {(c.base.suit, c.base.rank) for c in gs["deck"]}
        assert len(gs["deck"]) == 52
        assert len(identities) == 1  # the famous all-one-card deck
