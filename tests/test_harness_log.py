"""harness.log emission tests (output.md §6.1).

Verifies the lifecycle events appear in run_dir/logs/harness.log with
the expected fields, and that held-out seed values never leak."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.harness_log import HarnessLog, _format_kv  # noqa: E402
from hlbench.core.server import Server  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_POLICY = _REPO_ROOT / "agents" / "pd_pendulum" / "policy.py"


# --------------------------- pure helpers --------------------------------


def test_format_kv_quotes_strings_with_whitespace() -> None:
    assert _format_kv({"a": "x y"}) == 'a="x y"'
    assert _format_kv({"a": "plain"}) == "a=plain"
    assert _format_kv({"a": 1, "b": 1.5, "c": True, "d": None}) == \
        "a=1 b=1.5 c=True d=None"


def test_format_kv_escapes_quotes() -> None:
    assert _format_kv({"msg": 'has "quotes" in it'}) == 'msg="has \\"quotes\\" in it"'


def test_harness_log_disabled_is_noop(tmp_path: Path) -> None:
    """disabled() writer accepts all events without raising or writing."""
    log = HarnessLog.disabled()
    log.event("anything", foo=1, bar="baz")
    assert log.path is None
    assert not any(tmp_path.iterdir())  # nothing written anywhere


def test_harness_log_swallows_write_errors(tmp_path: Path) -> None:
    """If the file disappears mid-run, .event() must not raise."""
    log_path = tmp_path / "harness.log"
    log = HarnessLog(log_path)
    log.event("first", x=1)
    # Replace the parent dir with a regular file → subsequent writes fail
    # at the OS level. The contextlib.suppress in event() swallows it.
    shutil.rmtree(tmp_path)
    tmp_path.write_text("now a file")
    log.event("second", x=2)  # must not raise


# --------------------------- Server integration --------------------------


def _parse_event_line(line: str) -> tuple[str, str, dict[str, str]]:
    """Returns (level, event_name, kv_dict). Reverses _format_kv well
    enough for test assertions."""
    m = re.match(
        r"(?P<ts>\S+)\s+(?P<level>\S+)\s+(?P<name>\S+)(?:\s+(?P<rest>.*))?$",
        line,
    )
    assert m, f"unparseable: {line!r}"
    rest = m.group("rest") or ""
    kv: dict[str, str] = {}
    # Simple tokenizer: split on spaces NOT inside quotes.
    tokens = re.findall(r'(?:[^\s"]+|"(?:\\.|[^"\\])*")+', rest)
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
        kv[k] = v
    return m.group("level"), m.group("name"), kv


def _read_events(log_path: Path) -> list[tuple[str, dict[str, str]]]:
    """Parse harness.log → list of (event_name, kv_dict). Drops level."""
    return [
        (name, kv)
        for (_level, name, kv) in (
            _parse_event_line(line) for line in log_path.read_text().splitlines()
            if line.strip()
        )
    ]


@pytest.fixture
def server_with_log(tmp_path: Path):
    srv = Server(env_id="pendulum", runs_root=tmp_path / "runs")
    shutil.copy(_REFERENCE_POLICY, srv.workspace_dir / "system" / "policy.py")
    return srv


def test_run_start_event_written_on_server_init(server_with_log: Server) -> None:
    log_path = server_with_log.run_dir / "logs" / "harness.log"
    assert log_path.is_file()
    events = _read_events(log_path)
    names = [name for (name, _) in events]
    assert names[0] == "run_start"
    _, kv = events[0]
    assert kv["env"] == "pendulum"
    assert "episode_budget" in kv
    assert "harness_version" in kv


def test_submit_and_finalize_lifecycle_events(server_with_log: Server) -> None:
    srv = server_with_log
    srv.submit([0, 1, 2])
    srv.finalize()
    log_path = srv.run_dir / "logs" / "harness.log"
    events = _read_events(log_path)
    names = [name for (name, _) in events]

    # Expected sequence (with episodes between submit_received and
    # submit_completed).
    assert names[0] == "run_start"
    assert "submit_received" in names
    assert "snapshot_taken" in names
    assert names.count("episode_start") == 3
    assert names.count("episode_end") == 3
    assert "submit_completed" in names
    assert "finalize_start" in names
    assert names[-1] == "run_end"

    # run_end carries the final_score (or 'n/a' on error).
    _, end_kv = events[-1]
    assert end_kv["status"] in ("completed", "error", "aborted")
    assert "final_score" in end_kv
    assert "wall_time_seconds" in end_kv


def test_log_never_contains_real_seed_values(server_with_log: Server) -> None:
    """CLAUDE.md invariant 3: real seeds are server-internal. Even at
    the per-episode log level, only the env_instance ID is recorded."""
    srv = server_with_log
    srv.submit([0, 5, 10])
    srv.finalize()

    log_text = (srv.run_dir / "logs" / "harness.log").read_text()
    # Read the actual real seeds the env would have used, verify NONE
    # appear in the log (the file is small enough to substring-scan).
    from hlbench.core.seed_manager import SeedManager
    from hlbench.envs.registry import get_env
    sm = SeedManager(
        get_env("pendulum").train_seeds_path,
        get_env("pendulum").heldout_seeds_path,
    )
    for env_inst in (0, 5, 10):
        real_seed = sm.real_seed_for_instance(env_inst)
        # Use a word boundary so accidental substring matches (e.g. an
        # ID == prefix of seed) don't false-positive.
        assert not re.search(rf"\b{real_seed}\b", log_text), (
            f"real seed {real_seed} for env_instance {env_inst} leaked "
            "into harness.log"
        )
    # Held-out seeds (the agent never sees these in any form).
    for held_out_seed in sm.held_out_seeds():
        assert not re.search(rf"\b{held_out_seed}\b", log_text), (
            f"held-out seed {held_out_seed} leaked into harness.log"
        )


def test_run_json_artifacts_points_at_harness_log(server_with_log: Server) -> None:
    srv = server_with_log
    srv.submit([0])
    final = srv.finalize()
    import json
    doc = json.loads(final.run_json_path.read_text())
    assert doc["artifacts"]["logs_harness"] == "logs/harness.log"
