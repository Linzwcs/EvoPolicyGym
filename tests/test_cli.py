"""CLI tests — end-to-end via the argparse functions.

Drives ``hlbench_cli.main`` directly (no subprocess) and points the
HTTP-using subcommands at a real HlbenchHTTPServer running in a
background thread."""

from __future__ import annotations

import io
import json
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.server import Server  # noqa: E402
from hlbench.http_server import HlbenchHTTPServer  # noqa: E402
from hlbench_cli.main import _build_parser, _parse_env_instances, main  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_POLICY = _REPO_ROOT / "agents" / "pd_pendulum" / "policy.py"


# --------------------------- pure helpers --------------------------------


def test_parse_env_instances_range() -> None:
    assert _parse_env_instances("0-3") == [0, 1, 2, 3]


def test_parse_env_instances_list() -> None:
    assert _parse_env_instances("0,2,5") == [0, 2, 5]


def test_parse_env_instances_mix() -> None:
    assert _parse_env_instances("0-2,7,10-11") == [0, 1, 2, 7, 10, 11]


# --------------------------- cmd_init ------------------------------------


def test_init_creates_workspace(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    rc = main([
        "init", "--env", "pendulum",
        "--runs-root", str(runs_root),
        "--model", "test-model",
        "--exp-id", "myrun",
    ])
    assert rc == 0
    # Canonical layout: runs_root/<model>/<env>/<exp-id>/workspace/
    ws = runs_root / "test-model" / "pendulum" / "myrun" / "workspace"
    assert (ws / "AGENTS.md").is_file()
    assert (ws / "system").is_dir()
    assert (ws / "feedback").is_dir()
    # TASK.md is NOT staged into workspace (served via GET /task).
    assert not (ws / "TASK.md").exists()
    # And the sibling dirs (checkpoints/, logs/) are created at run_dir.
    run_dir = ws.parent
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "logs").is_dir()


# --------------------------- HTTP-using subcommands ----------------------


@pytest.fixture
def live_server(tmp_path: Path):
    srv = Server(env_id="pendulum", runs_root=tmp_path / "runs")
    shutil.copy(_REFERENCE_POLICY, srv.workspace_dir / "system" / "policy.py")
    with HlbenchHTTPServer(srv, port=0) as http:
        yield f"http://{http.host}:{http.port}"


def test_info_command_prints_state(live_server) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["info", "--url", live_server])
    assert rc == 0
    out = buf.getvalue()
    assert "env:" in out and "pendulum" in out
    assert "budget:" in out
    assert "256" in out  # full budget at start


def test_info_raw_emits_full_json(live_server) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["info", "--url", live_server, "--raw"])
    parsed = json.loads(buf.getvalue())
    assert parsed["env"] == "pendulum"
    assert "state" in parsed


def test_submit_command_returns_zero_on_ok(live_server) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "submit", "--env-instances", "0-3", "--url", live_server,
        ])
    assert rc == 0
    out = buf.getvalue()
    assert "submit #0: ok" in out
    assert "mean_return" in out


def test_submit_command_returns_one_on_failure(live_server) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([
            "submit", "--env-instances", "999", "--url", live_server,
        ])
    assert rc == 1  # invalid_env_instance → status != "ok"
    assert "invalid_env_instance" in buf.getvalue()


def test_finalize_command_succeeds_after_submit(live_server) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["submit", "--env-instances", "0-1", "--url", live_server])
        rc = main(["finalize", "--url", live_server])
    assert rc == 0
    out = buf.getvalue()
    assert "finalize: completed" in out
    assert "final_score" in out


def test_unreachable_server_exits_with_error(tmp_path: Path) -> None:
    """When the server isn't running, CLI should fail fast (exit 2),
    not hang or raise."""
    with pytest.raises(SystemExit) as exc:
        main(["info", "--url", "http://127.0.0.1:1"])  # likely-unused port
    assert exc.value.code == 2


# --------------------------- argparse --------------------------------------


def test_parser_requires_subcommand() -> None:
    p = _build_parser()
    with pytest.raises(SystemExit):
        # No args → argparse exits because subparser is required.
        p.parse_args([])
