"""HTTP wrapper tests — spin up HlbenchHTTPServer in a background thread
and hit it with urllib. No external HTTP client dependency."""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import pytest

gym = pytest.importorskip("gymnasium")

from hlbench.core.server import Server  # noqa: E402
from hlbench.http_server import HlbenchHTTPServer  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_POLICY = _REPO_ROOT / "agents" / "pd_pendulum" / "policy.py"


@pytest.fixture
def http_server(tmp_path: Path):
    """Yield (base_url, server) for a live HTTP wrapper around a fresh
    Pendulum Server with the reference PD agent staged."""
    srv = Server(env_id="pendulum", runs_root=tmp_path / "runs")
    shutil.copy(_REFERENCE_POLICY, srv.workspace_dir / "system" / "policy.py")
    # port=0 → OS picks a free port; HTTP server reports the real one.
    with HlbenchHTTPServer(srv, port=0) as http:
        yield f"http://{http.host}:{http.port}", srv


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------- GET /info ------------------------------------


def test_health_endpoint_returns_ok(http_server) -> None:
    url, _ = http_server
    assert _get(f"{url}/health") == {"ok": True}


def test_task_endpoint_returns_markdown(http_server) -> None:
    """GET /task returns the env's TASK.md as text/markdown (not JSON)."""
    url, _ = http_server
    with urllib.request.urlopen(f"{url}/task", timeout=5) as resp:
        ctype = resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
    assert ctype.startswith("text/markdown"), f"unexpected Content-Type: {ctype!r}"
    assert "Pendulum" in body, "Pendulum env-specific TASK.md should be served"
    # Body is raw markdown, not JSON-wrapped.
    assert not body.lstrip().startswith("{"), "should not be JSON envelope"


def test_info_returns_spec_schema(http_server) -> None:
    url, _ = http_server
    info = _get(f"{url}/info")
    assert info["env"] == "pendulum"
    assert info["schema_version"] == "0.1"
    assert info["state"]["remaining_budget"] == 256
    # Held-out details and baselines remain hidden via HTTP too.
    flat = json.dumps(info)
    assert "expert_baseline" not in flat
    assert "random_baseline" not in flat
    assert "real_seed" not in flat


# --------------------------- POST /submit ---------------------------------


def test_submit_then_info_shows_updated_state(http_server) -> None:
    url, _ = http_server
    result = _post(f"{url}/submit", {"env_instances": [0, 1, 2, 3]})
    assert result["status"] == "ok"
    assert result["submit_id"] == 0
    assert result["summary"]["mean_return"] < 0

    info = _get(f"{url}/info")
    assert info["state"]["remaining_budget"] == 252
    assert info["state"]["n_submits"] == 1
    assert info["state"]["last_submit_status"] == "ok"


def test_submit_bad_body_returns_400(http_server) -> None:
    url, _ = http_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{url}/submit", {"wrong_key": [0]})
    assert exc.value.code == 400
    err = json.loads(exc.value.read().decode("utf-8"))
    assert err["error"] == "bad_request"


def test_submit_propagates_verdict_in_summary(http_server) -> None:
    """Phase 1 failures (invalid_env_instance) still return 200 — the
    failure rides on summary.status per SPEC §4.1."""
    url, _ = http_server
    # 99999 is above the 10000-id Pendulum train pool ⇒ invalid_env_instance.
    result = _post(f"{url}/submit", {"env_instances": [0, 99999]})
    assert result["status"] == "invalid_env_instance"
    assert result["summary"]["remaining_budget"] == 256  # Phase 1: no consume


# --------------------------- POST /finalize -------------------------------


def test_finalize_returns_run_json_body(http_server) -> None:
    url, _ = http_server
    _post(f"{url}/submit", {"env_instances": [0, 1]})
    final = _post(f"{url}/finalize")
    assert final["status"] == "completed"
    assert final["final_score"] is not None and final["final_score"] > 0
    assert len(final["held_out_returns"]) == 256
    assert final["final_submit_index"] == 0
    # run_json_path was serialized via the Path encoder.
    assert "run.json" in final["run_json_path"]


def test_submit_after_finalize_returns_409(http_server) -> None:
    url, _ = http_server
    _post(f"{url}/submit", {"env_instances": [0]})
    _post(f"{url}/finalize")
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{url}/submit", {"env_instances": [1]})
    assert exc.value.code == 409


# --------------------------- routing --------------------------------------


def test_unknown_path_returns_404(http_server) -> None:
    url, _ = http_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{url}/does/not/exist")
    assert exc.value.code == 404


def test_get_on_submit_returns_404(http_server) -> None:
    """We use POST for state-changing endpoints; GET should 404 (not 405)
    since /submit isn't a GET resource at all."""
    url, _ = http_server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{url}/submit")
    assert exc.value.code == 404
