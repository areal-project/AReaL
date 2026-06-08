"""Unit tests for RPCGuard Flask app (areal.experimental.inference_service.guard.app).

Tests all 4 endpoints (/health, /alloc_ports, /fork, /kill_forked_worker)
and the cleanup_forked_children() function using Flask test client with
mocked subprocess spawning.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from areal.experimental.inference_service.guard import app as guard_module
from areal.experimental.inference_service.guard.app import app, cleanup_forked_children

GUARD_APP = "areal.infra.rpc.guard.app"


@pytest.fixture(autouse=True)
def _reset_guard_globals():
    guard_module._state.allocated_ports = set()
    guard_module._state.forked_children = []
    guard_module._state.forked_children_map = {}
    guard_module._state.server_host = "10.0.0.1"
    guard_module._state.experiment_name = "test-exp"
    guard_module._state.trial_name = "test-trial"
    guard_module._state.fileroot = None
    yield
    guard_module._state.allocated_ports = set()
    guard_module._state.forked_children = []
    guard_module._state.forked_children_map = {}


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_mock_process(pid: int = 12345, running: bool = True) -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = pid
    proc.poll.return_value = None if running else 0
    return proc


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["forked_children"] == 0

    def test_health_counts_forked_children(self, client):
        guard_module._state.forked_children = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        resp = client.get("/health")
        data = resp.get_json()
        assert data["forked_children"] == 3


class TestAllocPorts:
    @patch(f"{GUARD_APP}.find_free_ports")
    def test_alloc_ports_success(self, mock_find, client):
        mock_find.return_value = [9001, 9002, 9003]
        resp = client.post("/alloc_ports", json={"count": 3})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["ports"] == [9001, 9002, 9003]
        assert data["host"] == "10.0.0.1"
        assert guard_module._state.allocated_ports == {9001, 9002, 9003}

    @patch(f"{GUARD_APP}.find_free_ports")
    def test_alloc_ports_excludes_previous(self, mock_find, client):
        mock_find.return_value = [9001, 9002, 9003]
        client.post("/alloc_ports", json={"count": 3})

        mock_find.return_value = [9004, 9005]
        resp = client.post("/alloc_ports", json={"count": 2})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ports"] == [9004, 9005]

        _, kwargs = mock_find.call_args
        assert 9001 in kwargs.get("exclude_ports", set())
        assert 9002 in kwargs.get("exclude_ports", set())
        assert 9003 in kwargs.get("exclude_ports", set())

        assert guard_module._state.allocated_ports == {
            9001,
            9002,
            9003,
            9004,
            9005,
        }

    def test_alloc_ports_missing_count(self, client):
        resp = client.post("/alloc_ports", json={})
        assert resp.status_code == 400
        assert "count" in resp.get_json()["error"].lower()

    def test_alloc_ports_invalid_count_zero(self, client):
        resp = client.post("/alloc_ports", json={"count": 0})
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count_negative(self, client):
        resp = client.post("/alloc_ports", json={"count": -1})
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count_string(self, client):
        resp = client.post("/alloc_ports", json={"count": "three"})
        assert resp.status_code == 400

    def test_alloc_ports_no_json_body(self, client):
        resp = client.post("/alloc_ports", data="not json", content_type="text/plain")
        assert resp.status_code == 400


class TestFork:
    @patch(f"{GUARD_APP}.run_with_streaming_logs")
    def test_fork_raw_cmd_passes_command_as_is(self, mock_run, client):
        mock_proc = _make_mock_process(pid=55)
        mock_run.return_value = mock_proc

        raw = [
            "python",
            "-m",
            "areal.experimental.inference_service.sglang.launch_server",
            "--model",
            "test-model",
        ]
        resp = client.post(
            "/fork",
            json={"role": "sglang", "worker_index": 0, "raw_cmd": raw},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert data["host"] == "10.0.0.1"
        assert data["pid"] == 55
        assert "port" not in data

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd == raw

    @patch(f"{GUARD_APP}.run_with_streaming_logs")
    def test_fork_tracks_child(self, mock_run, client):
        mock_proc = _make_mock_process(pid=42)
        mock_run.return_value = mock_proc

        client.post(
            "/fork",
            json={
                "role": "test",
                "worker_index": 0,
                "raw_cmd": ["echo", "hello"],
            },
        )

        assert mock_proc in guard_module._state.forked_children
        assert ("test", 0) in guard_module._state.forked_children_map
        assert guard_module._state.forked_children_map[("test", 0)] is mock_proc

    @patch(f"{GUARD_APP}.run_with_streaming_logs")
    def test_fork_with_env_overrides(self, mock_run, client):
        mock_run.return_value = _make_mock_process()

        resp = client.post(
            "/fork",
            json={
                "role": "test",
                "worker_index": 0,
                "raw_cmd": ["echo", "hello"],
                "env": {"MY_VAR": "my_value"},
            },
        )
        assert resp.status_code == 200

        call_kwargs = mock_run.call_args
        child_env = call_kwargs[1]["env"]
        assert child_env["MY_VAR"] == "my_value"


class TestForkErrorHandling:
    def test_fork_missing_role(self, client):
        resp = client.post(
            "/fork",
            json={"worker_index": 0, "raw_cmd": ["echo"]},
        )
        assert resp.status_code == 400
        assert "role" in resp.get_json()["error"].lower()

    def test_fork_missing_worker_index(self, client):
        resp = client.post(
            "/fork",
            json={"role": "test", "raw_cmd": ["echo"]},
        )
        assert resp.status_code == 400
        assert "worker_index" in resp.get_json()["error"].lower()

    def test_fork_missing_raw_cmd(self, client):
        resp = client.post("/fork", json={"role": "test", "worker_index": 0})
        assert resp.status_code == 400
        assert "raw_cmd" in resp.get_json()["error"].lower()

    def test_fork_no_json_body(self, client):
        resp = client.post("/fork", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_alloc_ports_invalid_count(self, client):
        resp = client.post("/alloc_ports", json={"count": 1.5})
        assert resp.status_code == 400


class TestKillForkedWorker:
    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_kill_known_worker(self, mock_kill, client):
        mock_proc = _make_mock_process(pid=123)
        guard_module._state.forked_children.append(mock_proc)
        guard_module._state.forked_children_map[("test", 0)] = mock_proc

        resp = client.post(
            "/kill_forked_worker",
            json={"role": "test", "worker_index": 0},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "123" in data["message"]

        assert mock_proc not in guard_module._state.forked_children
        assert ("test", 0) not in guard_module._state.forked_children_map

        mock_kill.assert_called_once_with(123, timeout=3, graceful=True)

    def test_kill_unknown_worker_returns_404(self, client):
        resp = client.post(
            "/kill_forked_worker",
            json={"role": "ghost", "worker_index": 99},
        )
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_kill_already_exited_worker(self, mock_kill, client):
        mock_proc = _make_mock_process(pid=456, running=False)
        guard_module._state.forked_children.append(mock_proc)
        guard_module._state.forked_children_map[("done", 0)] = mock_proc

        resp = client.post(
            "/kill_forked_worker",
            json={"role": "done", "worker_index": 0},
        )
        assert resp.status_code == 200
        mock_kill.assert_not_called()

    def test_kill_missing_role(self, client):
        resp = client.post("/kill_forked_worker", json={"worker_index": 0})
        assert resp.status_code == 400
        assert "role" in resp.get_json()["error"].lower()

    def test_kill_missing_worker_index(self, client):
        resp = client.post("/kill_forked_worker", json={"role": "test"})
        assert resp.status_code == 400
        assert "worker_index" in resp.get_json()["error"].lower()

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_kill_then_kill_again_returns_404(self, mock_kill, client):
        mock_proc = _make_mock_process(pid=789)
        guard_module._state.forked_children.append(mock_proc)
        guard_module._state.forked_children_map[("test", 0)] = mock_proc

        resp1 = client.post(
            "/kill_forked_worker",
            json={"role": "test", "worker_index": 0},
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            "/kill_forked_worker",
            json={"role": "test", "worker_index": 0},
        )
        assert resp2.status_code == 404


class TestCleanup:
    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_kills_all_running_children(self, mock_kill):
        proc1 = _make_mock_process(pid=100)
        proc2 = _make_mock_process(pid=200)
        guard_module._state.forked_children = [proc1, proc2]
        guard_module._state.forked_children_map = {
            ("a", 0): proc1,
            ("b", 0): proc2,
        }

        cleanup_forked_children()

        assert mock_kill.call_count == 2
        pids_killed = {call.args[0] for call in mock_kill.call_args_list}
        assert pids_killed == {100, 200}

        assert guard_module._state.forked_children == []
        assert guard_module._state.forked_children_map == {}

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_skips_already_exited(self, mock_kill):
        running = _make_mock_process(pid=100, running=True)
        exited = _make_mock_process(pid=200, running=False)
        guard_module._state.forked_children = [running, exited]
        guard_module._state.forked_children_map = {
            ("a", 0): running,
            ("b", 0): exited,
        }

        cleanup_forked_children()

        mock_kill.assert_called_once_with(100, timeout=3, graceful=True)

        assert guard_module._state.forked_children == []
        assert guard_module._state.forked_children_map == {}

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_no_children_is_noop(self, mock_kill):
        cleanup_forked_children()
        mock_kill.assert_not_called()

    @patch(f"{GUARD_APP}.kill_process_tree")
    def test_cleanup_tolerates_kill_exception(self, mock_kill):
        proc1 = _make_mock_process(pid=100)
        proc2 = _make_mock_process(pid=200)
        guard_module._state.forked_children = [proc1, proc2]
        guard_module._state.forked_children_map = {
            ("a", 0): proc1,
            ("b", 0): proc2,
        }

        mock_kill.side_effect = [OSError("boom"), None]

        cleanup_forked_children()

        assert mock_kill.call_count == 2
        assert guard_module._state.forked_children == []
        assert guard_module._state.forked_children_map == {}
