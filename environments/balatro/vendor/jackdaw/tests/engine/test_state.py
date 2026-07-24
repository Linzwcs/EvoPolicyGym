"""Tests for jackdaw.engine.state."""

from __future__ import annotations

from jackdaw.engine.state import describe_state


def test_describe_state_basic():
    """describe_state returns a readable one-line summary."""
    gs = {
        "phase": "selecting_hand",
        "round_resets": {"ante": 2},
        "dollars": 12,
        "hand": [None] * 8,
        "jokers": [None] * 3,
        "chips": 0,
        "blind": None,
    }
    result = describe_state(gs)
    assert "selecting_hand" in result
    assert "ante=2" in result
    assert "$12" in result
    assert "hand=8" in result
    assert "jokers=3" in result
    assert "chips=0" in result


def test_describe_state_empty():
    """describe_state handles a minimal/empty dict without crashing."""
    result = describe_state({})
    assert "?" in result  # phase defaults to "?"
