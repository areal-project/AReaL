from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from areal.infra.data_service.guard.app import (
    GuardState,
    cleanup_forked_children,
    create_app,
)


@pytest.fixture()
def state() -> GuardState:
    s = GuardState()
    s.server_host = "10.0.0.1"
    s.experiment_name = "test-exp"
    s.trial_name = "test-trial"
    s.role = "test-role"
    s.worker_index = 0
    return s


@pytest.fixture()
def client(state: GuardState):
    app = create_app(state)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _make_mock_process(pid: int = 12345, running: bool = True) -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = pid
    proc.poll.return_value = None if running else 0
    return proc


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert data["forked_children"] == 0


@patch("areal.infra.rpc.guard.app.find_free_ports")
def test_alloc_ports_success(mock_find, client, state: GuardState):
    mock_find.return_value = [9001, 9002]
    resp = client.post("/alloc_ports", json={"count": 2})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ports"] == [9001, 9002]
    assert data["host"] == "10.0.0.1"
    assert state.allocated_ports == {9001, 9002}


@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
def test_fork_raw_command_success(mock_run, client, state: GuardState):
    mock_proc = _make_mock_process(pid=42)
    mock_run.return_value = mock_proc

    resp = client.post(
        "/fork",
        json={
            "role": "worker",
            "worker_index": 1,
            "raw_cmd": ["python", "-m", "module", "--port", "8001"],
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["host"] == "10.0.0.1"
    assert data["pid"] == 42
    assert ("worker", 1) in state.forked_children_map


@patch("areal.infra.rpc.guard.app.kill_process_tree")
def test_kill_known_worker(mock_kill, client, state: GuardState):
    mock_proc = _make_mock_process(pid=123)
    state.forked_children.append(mock_proc)
    state.forked_children_map[("test", 0)] = mock_proc

    resp = client.post("/kill_forked_worker", json={"role": "test", "worker_index": 0})
    assert resp.status_code == 200
    assert ("test", 0) not in state.forked_children_map
    mock_kill.assert_called_once_with(123, timeout=3, graceful=True)


@patch("areal.infra.rpc.guard.app.kill_process_tree")
def test_cleanup_kills_all_running_children(mock_kill, state: GuardState):
    proc1 = _make_mock_process(pid=100)
    proc2 = _make_mock_process(pid=200)
    state.forked_children = [proc1, proc2]
    state.forked_children_map = {("a", 0): proc1, ("b", 0): proc2}

    cleanup_forked_children(state)

    assert mock_kill.call_count == 2
    assert state.forked_children == []
    assert state.forked_children_map == {}
