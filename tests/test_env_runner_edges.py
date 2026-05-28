"""Edge-case tests for env_runner + feedback that don't fire on Pendulum.

Pendulum has empty info dicts and clean float rewards, so the
trajectory serialization paths for numpy info entries and NaN/Inf
encoding never get exercised by the main e2e tests. These verify
SPEC §4.2 invariants directly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hlbench.core.env_runner import EpisodeRecord, _info_to_jsonable
from hlbench.core.feedback import _jsonify_floats, write_trajectory

# --------------------------- _info_to_jsonable ----------------------------


def test_info_to_jsonable_passthrough_for_python_types() -> None:
    info = {"score": 3, "name": "abc", "ok": True, "ratio": 0.5}
    assert _info_to_jsonable(info) == info


def test_info_to_jsonable_unwraps_numpy_arrays_and_scalars() -> None:
    np = pytest.importorskip("numpy")
    info = {
        "arr": np.array([1.0, 2.0, 3.0]),
        "scalar": np.float64(7.5),
        "int_scalar": np.int32(42),
        "plain_int": 99,  # mixed: should pass through
    }
    out = _info_to_jsonable(info)
    assert out["arr"] == [1.0, 2.0, 3.0]
    assert out["scalar"] == 7.5
    assert out["int_scalar"] == 42
    assert out["plain_int"] == 99
    # The output must be JSON-serializable end-to-end.
    json.dumps(out)


# --------------------------- NaN / Inf encoding ---------------------------


def test_jsonify_floats_encodes_nan_inf_as_strings() -> None:
    """SPEC §4.2: JSON has no NaN/Inf literals; we encode as 'NaN'/'Inf'/'-Inf'."""
    assert _jsonify_floats(float("nan")) == "NaN"
    assert _jsonify_floats(float("inf")) == "Inf"
    assert _jsonify_floats(float("-inf")) == "-Inf"


def test_jsonify_floats_recurses_into_dict_and_list() -> None:
    out = _jsonify_floats({
        "ok": 1.5,
        "weird": [float("nan"), float("inf"), {"deep": float("-inf")}],
    })
    assert out == {
        "ok": 1.5,
        "weird": ["NaN", "Inf", {"deep": "-Inf"}],
    }


def test_jsonify_floats_leaves_other_types_alone() -> None:
    sentinel = object()
    assert _jsonify_floats(sentinel) is sentinel
    assert _jsonify_floats("string") == "string"
    assert _jsonify_floats(42) == 42  # int, not float — passthrough


def test_write_trajectory_handles_nan_reward(tmp_path: Any) -> None:
    """End-to-end: a step with NaN reward must produce valid JSONL where
    the reward field is the string ``"NaN"``."""
    path = tmp_path / "traj.jsonl"
    entry = {
        "t": 0, "obs": [0.0], "action": [0.0],
        "reward": float("nan"),
        "terminated": False, "truncated": False, "info": {},
    }
    write_trajectory(path, [entry])
    line = path.read_text().strip()
    parsed = json.loads(line)
    assert parsed["reward"] == "NaN"
    # And it's a single line (no embedded newlines).
    assert "\n" not in line


def test_write_trajectory_empty_list_writes_empty_file(tmp_path: Any) -> None:
    """Edge: zero-step episode (e.g. immediate reset error). The file
    exists but is empty — readers MUST tolerate this since reset_error
    triggers it."""
    path = tmp_path / "traj.jsonl"
    write_trajectory(path, [])
    assert path.read_text() == ""


# --------------------------- on_episode_end_error ------------------------


class _GoodPolicy:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def reset(self, episode_index: int) -> None:
        pass

    def act(self, obs: Any) -> Any:
        return [0.0]

    def on_episode_end(self, episode_return: float) -> None:
        raise RuntimeError("boom in on_episode_end")


class _CrashAtStep3:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.calls = 0

    def reset(self, episode_index: int) -> None:
        self.calls = 0

    def act(self, obs: Any) -> Any:
        self.calls += 1
        if self.calls > 3:
            raise ValueError("act crash at step 4")
        return [0.0]

    def on_episode_end(self, episode_return: float) -> None:
        raise RuntimeError("on_episode_end also crashes")


def test_on_episode_end_error_recorded_when_act_succeeded() -> None:
    """No act error → on_episode_end raise is what classifies the
    episode."""
    gym = pytest.importorskip("gymnasium")
    env = gym.make("Pendulum-v1")
    try:
        from hlbench.core.env_runner import run_episode

        rec = run_episode(
            _GoodPolicy(),
            env,
            real_seed=0,
            episode_index=0,
            action_space_type="Box",
            max_steps=5,  # finishes via max_steps → on_episode_end fires
        )
        assert rec.length == 5
        assert rec.ended_with_error is True
        assert rec.error_category == "on_episode_end_error"
        assert rec.error_step_index == 5  # len(trajectory)
        assert "boom in on_episode_end" in (rec.error_traceback or "")
    finally:
        env.close()


def test_on_episode_end_error_does_not_clobber_earlier_act_error() -> None:
    """Both fail → act_error wins because it's what actually ended the
    episode. on_episode_end's failure is silently absorbed (still
    ended_with_error=True, but the act traceback is what we surface)."""
    gym = pytest.importorskip("gymnasium")
    env = gym.make("Pendulum-v1")
    try:
        from hlbench.core.env_runner import run_episode

        rec = run_episode(
            _CrashAtStep3(),
            env,
            real_seed=0,
            episode_index=0,
            action_space_type="Box",
            max_steps=20,
        )
        assert rec.ended_with_error is True
        assert rec.error_category == "act_error"
        assert "act crash at step 4" in (rec.error_traceback or "")
        # The on_episode_end error did NOT overwrite the act traceback.
        assert "on_episode_end also crashes" not in (rec.error_traceback or "")
    finally:
        env.close()


# --------------------------- EpisodeRecord defaults ---------------------


def test_episode_record_defaults_for_success() -> None:
    """A clean record carries None error fields — used by submit_handler
    to decide whether to write per-episode error.txt."""
    rec = EpisodeRecord(
        trajectory=[], return_=0.0, length=0,
        terminated=True, truncated=False,
    )
    assert rec.ended_with_error is False
    assert rec.error_category is None
    assert rec.error_step_index is None
    assert rec.error_traceback is None
