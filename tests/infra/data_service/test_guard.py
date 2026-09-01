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
    yield s
    for lock_file in s.port_lock_files.values():
        lock_file.close()


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


@patch("areal.infra.rpc.guard.app.find_free_ports")
def test_owned_ports_release_after_failed_fork(mock_find, client, state: GuardState):
    mock_find.return_value = [9010]
    alloc = client.post(
        "/alloc_ports",
        json={"count": 1, "role": "worker", "worker_index": 2},
    )
    assert alloc.status_code == 200

    released = client.post("/release_ports", json={"role": "worker", "worker_index": 2})

    assert released.status_code == 200
    assert released.get_json()["ports"] == [9010]
    assert 9010 not in state.allocated_ports
    assert ("worker", 2) not in state.owned_ports


@patch("areal.infra.rpc.guard.app.find_free_ports")
def test_alloc_ports_duplicate_owner_returns_conflict(
    mock_find, client, state: GuardState
):
    mock_find.return_value = [9011]
    payload = {"count": 1, "role": "worker", "worker_index": 3}

    first = client.post("/alloc_ports", json=payload)
    second = client.post("/alloc_ports", json=payload)

    assert first.status_code == 200
    assert second.status_code == 409
    assert state.owned_ports[("worker", 3)] == {9011}


def test_release_ports_running_child_returns_conflict(client, state: GuardState):
    mock_proc = _make_mock_process(pid=41)
    state.forked_children_map[("worker", 4)] = mock_proc
    state.owned_ports[("worker", 4)] = {9012}
    state.allocated_ports.add(9012)

    response = client.post("/release_ports", json={"role": "worker", "worker_index": 4})

    assert response.status_code == 409
    assert state.owned_ports[("worker", 4)] == {9012}
    assert state.allocated_ports == {9012}


@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
def test_fork_without_port_reservation_returns_conflict(
    mock_run, client, state: GuardState
):
    response = client.post(
        "/fork",
        json={
            "role": "worker",
            "worker_index": 5,
            "raw_cmd": ["python", "-m", "module"],
        },
    )

    assert response.status_code == 409
    mock_run.assert_not_called()


@patch(
    "areal.infra.rpc.guard.app.run_with_streaming_logs",
    side_effect=RuntimeError("spawn failed"),
)
def test_fork_spawn_failure_releases_owned_ports(mock_run, client, state: GuardState):
    state.owned_ports[("worker", 6)] = {9013}
    state.allocated_ports.add(9013)

    response = client.post(
        "/fork",
        json={
            "role": "worker",
            "worker_index": 6,
            "raw_cmd": ["python", "-m", "module", "--port", "9013"],
        },
    )

    assert response.status_code == 500
    assert ("worker", 6) not in state.owned_ports
    assert 9013 not in state.allocated_ports


@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
def test_fork_raw_command_success(mock_run, client, state: GuardState):
    mock_proc = _make_mock_process(pid=42)
    mock_run.return_value = mock_proc
    state.owned_ports[("worker", 1)] = {8001}
    state.allocated_ports.add(8001)

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
    state.owned_ports[("test", 0)] = {9014}
    state.allocated_ports.add(9014)

    resp = client.post("/kill_forked_worker", json={"role": "test", "worker_index": 0})
    assert resp.status_code == 200
    assert resp.get_json()["released_ports"] == [9014]
    assert ("test", 0) not in state.forked_children_map
    assert ("test", 0) not in state.owned_ports
    assert 9014 not in state.allocated_ports
    mock_kill.assert_called_once_with(123, timeout=3, graceful=True)


@patch("areal.infra.rpc.guard.app.kill_process_tree", side_effect=RuntimeError("busy"))
def test_failed_kill_keeps_child_and_ports_for_retry(mock_kill, client, state):
    mock_proc = _make_mock_process(pid=124)
    state.forked_children.append(mock_proc)
    state.forked_children_map[("test", 1)] = mock_proc
    state.owned_ports[("test", 1)] = {9020}
    state.allocated_ports.add(9020)

    resp = client.post("/kill_forked_worker", json={"role": "test", "worker_index": 1})

    assert resp.status_code == 500
    assert state.forked_children_map[("test", 1)] is mock_proc
    assert state.owned_ports[("test", 1)] == {9020}


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
