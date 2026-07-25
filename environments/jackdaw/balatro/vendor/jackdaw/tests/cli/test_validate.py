"""Tests for the validation CLI — scenario framework and runner."""

from __future__ import annotations

from jackdaw.cli.scenarios import ScenarioResult, get_all_scenarios, get_scenarios


class TestScenarioRegistry:
    def test_get_all_scenarios_returns_list(self) -> None:
        scenarios = get_all_scenarios()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0

    def test_all_scenarios_have_required_fields(self) -> None:
        for s in get_all_scenarios():
            assert s.name, f"Scenario missing name: {s}"
            assert s.category, f"Scenario {s.name} missing category"
            assert s.description, f"Scenario {s.name} missing description"
            assert callable(s.run), f"Scenario {s.name} run is not callable"

    def test_all_scenario_names_unique(self) -> None:
        names = [s.name for s in get_all_scenarios()]
        assert len(names) == len(set(names)), (
            f"Duplicate scenario names: {[n for n in names if names.count(n) > 1]}"
        )

    def test_filter_by_category(self) -> None:
        jokers = get_scenarios(category="jokers")
        assert len(jokers) > 0
        assert all(s.category == "jokers" for s in jokers)

    def test_filter_by_name(self) -> None:
        results = get_scenarios(name="joker_joker")
        assert len(results) == 1
        assert results[0].name == "joker_joker"

    def test_filter_nonexistent_name(self) -> None:
        results = get_scenarios(name="nonexistent_scenario_xyz")
        assert len(results) == 0

    def test_categories_present(self) -> None:
        categories = {s.category for s in get_all_scenarios()}
        expected = {"jokers", "tarots", "planets", "spectrals", "boss_blinds", "modifiers"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"


class TestScenarioCoverage:
    """Verify comprehensive coverage of game mechanics."""

    def test_joker_count(self) -> None:
        """Should have a scenario for every joker."""
        jokers = get_scenarios(category="jokers")
        # 150 jokers total, some registered in bulk, some special
        assert len(jokers) >= 140, f"Only {len(jokers)} joker scenarios (expected ~150)"

    def test_tarot_count(self) -> None:
        tarots = get_scenarios(category="tarots")
        assert len(tarots) >= 20, f"Only {len(tarots)} tarot scenarios (expected ~22)"

    def test_planet_count(self) -> None:
        planets = get_scenarios(category="planets")
        assert len(planets) >= 13, f"Only {len(planets)} planet scenarios (expected 13)"

    def test_spectral_count(self) -> None:
        spectrals = get_scenarios(category="spectrals")
        assert len(spectrals) >= 15, f"Only {len(spectrals)} spectral scenarios (expected ~18)"

    def test_boss_blind_count(self) -> None:
        blinds = get_scenarios(category="boss_blinds")
        assert len(blinds) >= 5, (
            f"Only {len(blinds)} boss blind scenarios (expected ~7 seed groups)"
        )

    def test_boss_blind_coverage(self) -> None:
        """All 28 boss keys should be covered across seed groups."""
        from jackdaw.cli.scenarios.boss_blinds import _BOSS_SEEDS

        assert len(_BOSS_SEEDS) >= 25, (
            f"Only {len(_BOSS_SEEDS)} boss keys in _BOSS_SEEDS (expected ~28)"
        )

    def test_modifier_count(self) -> None:
        modifiers = get_scenarios(category="modifiers")
        assert len(modifiers) >= 15, f"Only {len(modifiers)} modifier scenarios (expected ~20)"


class TestScenarioResult:
    def test_passed_result(self) -> None:
        r = ScenarioResult(passed=True)
        assert r.passed
        assert r.diffs == []

    def test_failed_result(self) -> None:
        r = ScenarioResult(passed=False, diffs=["money: sim=10 live=12"])
        assert not r.passed
        assert len(r.diffs) == 1
