from __future__ import annotations

import subprocess
import threading
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


class _ObservableLock:
    def __init__(self, lock, thread_name: str, attempted: threading.Event):
        self._lock = lock
        self._thread_name = thread_name
        self._attempted = attempted

    def __enter__(self):
        if threading.current_thread().name == self._thread_name:
            self._attempted.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


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


@pytest.mark.parametrize("bad_index", ["bad", 1.5, True])
@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        ("/alloc_ports", {"count": 1, "role": "worker"}),
        ("/release_ports", {"role": "worker"}),
        ("/fork", {"role": "worker", "raw_cmd": ["python"]}),
        ("/kill_forked_worker", {"role": "worker"}),
    ],
)
def test_lifecycle_routes_invalid_worker_index_returns_bad_request(
    client, state: GuardState, endpoint: str, payload: dict, bad_index
):
    """Invalid owner indices are client errors and never mutate Guard state."""
    response = client.post(endpoint, json={**payload, "worker_index": bad_index})

    assert response.status_code == 400
    assert state.allocated_ports == set()
    assert state.owned_ports == {}
    assert state.forked_children_map == {}


@patch("areal.infra.rpc.guard.app.kill_process_tree")
@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
@patch("areal.infra.rpc.guard.app.find_free_ports")
def test_lifecycle_routes_normalize_numeric_string_worker_index(
    mock_find, mock_run, mock_kill, client, state: GuardState
):
    """All lifecycle routes map a numeric string to the same integer owner key."""
    mock_find.side_effect = [[9015], [9016]]
    mock_run.return_value = _make_mock_process(pid=43)

    alloc = client.post(
        "/alloc_ports",
        json={"count": 1, "role": "worker", "worker_index": "7"},
    )
    forked = client.post(
        "/fork",
        json={"role": "worker", "worker_index": "7", "raw_cmd": ["python"]},
    )
    killed = client.post(
        "/kill_forked_worker", json={"role": "worker", "worker_index": "7"}
    )
    second_alloc = client.post(
        "/alloc_ports",
        json={"count": 1, "role": "worker", "worker_index": "8"},
    )
    released = client.post(
        "/release_ports", json={"role": "worker", "worker_index": "8"}
    )

    assert [response.status_code for response in (alloc, forked, killed)] == [
        200,
        200,
        200,
    ]
    assert [second_alloc.status_code, released.status_code] == [200, 200]
    assert state.allocated_ports == set()
    assert state.owned_ports == {}
    assert state.forked_children_map == {}
    mock_kill.assert_called_once_with(43, timeout=3, graceful=True)


@patch("areal.infra.rpc.guard.app.fcntl.flock")
@patch("areal.infra.rpc.guard.app.Path.open")
@patch("areal.infra.rpc.guard.app.find_free_ports")
def test_alloc_ports_partial_lock_failure_rolls_back(
    mock_find, mock_open, mock_flock, client, state: GuardState
):
    """A failed multi-port reservation closes every lock opened by that call."""
    first_lock = MagicMock()
    second_lock = MagicMock()
    first_lock.fileno.return_value = 10
    second_lock.fileno.return_value = 11
    mock_find.return_value = [9017, 9018]
    mock_open.side_effect = [first_lock, second_lock]
    mock_flock.side_effect = [None, OSError("lock failed")]

    response = client.post(
        "/alloc_ports",
        json={"count": 2, "role": "worker", "worker_index": 9},
    )

    assert response.status_code == 500
    assert state.port_lock_files == {}
    assert state.allocated_ports == set()
    assert state.owned_ports == {}
    first_lock.close.assert_called_once()
    second_lock.close.assert_called_once()


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
def test_concurrent_forks_start_only_one_process(mock_run, state: GuardState):
    """Two requests for one owner cannot both spawn a child process."""
    app = create_app(state)
    app.config["TESTING"] = True
    state.owned_ports[("worker", 10)] = {9019}
    state.allocated_ports.add(9019)
    spawn_entered = threading.Event()
    allow_spawn = threading.Event()
    second_lock_attempted = threading.Event()
    statuses: list[int] = []
    state.forked_children_lock = _ObservableLock(
        state.forked_children_lock, "second-fork", second_lock_attempted
    )

    def _spawn(*args, **kwargs):
        spawn_entered.set()
        assert allow_spawn.wait(timeout=5)
        return _make_mock_process(pid=44)

    def _fork_request():
        with app.test_client() as thread_client:
            response = thread_client.post(
                "/fork",
                json={
                    "role": "worker",
                    "worker_index": 10,
                    "raw_cmd": ["python"],
                },
            )
            statuses.append(response.status_code)

    mock_run.side_effect = _spawn
    first = threading.Thread(target=_fork_request)
    second = threading.Thread(target=_fork_request, name="second-fork")

    first.start()
    assert spawn_entered.wait(timeout=5)
    second.start()
    assert second_lock_attempted.wait(timeout=5)
    allow_spawn.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(statuses) == [200, 409]
    assert mock_run.call_count == 1
    assert state.forked_children_map[("worker", 10)].pid == 44


@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
def test_failed_fork_releases_before_waiting_fork_proceeds(mock_run, state: GuardState):
    """A failed fork drops its reservation before a waiting fork validates."""
    app = create_app(state)
    app.config["TESTING"] = True
    state.owned_ports[("worker", 12)] = {9024}
    state.allocated_ports.add(9024)
    spawn_entered = threading.Event()
    allow_failure = threading.Event()
    second_lock_attempted = threading.Event()
    statuses: list[int] = []
    state.forked_children_lock = _ObservableLock(
        state.forked_children_lock, "second-fork", second_lock_attempted
    )

    def _fail_spawn(*args, **kwargs):
        spawn_entered.set()
        assert allow_failure.wait(timeout=5)
        raise RuntimeError("spawn failed")

    def _fork_request():
        with app.test_client() as thread_client:
            response = thread_client.post(
                "/fork",
                json={
                    "role": "worker",
                    "worker_index": 12,
                    "raw_cmd": ["python"],
                },
            )
            statuses.append(response.status_code)

    mock_run.side_effect = _fail_spawn
    first = threading.Thread(target=_fork_request)
    second = threading.Thread(target=_fork_request, name="second-fork")

    first.start()
    assert spawn_entered.wait(timeout=5)
    second.start()
    assert second_lock_attempted.wait(timeout=5)
    allow_failure.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(statuses) == [409, 500]
    assert mock_run.call_count == 1
    assert state.allocated_ports == set()
    assert state.owned_ports == {}
    assert state.forked_children_map == {}


@patch("areal.infra.rpc.guard.app.run_with_streaming_logs")
def test_release_waits_for_inflight_fork(mock_run, state: GuardState):
    """A release racing with Popen observes the registered child and conflicts."""
    app = create_app(state)
    app.config["TESTING"] = True
    state.owned_ports[("worker", 11)] = {9021}
    state.allocated_ports.add(9021)
    spawn_entered = threading.Event()
    allow_spawn = threading.Event()
    release_lock_attempted = threading.Event()
    statuses: dict[str, int] = {}
    state.forked_children_lock = _ObservableLock(
        state.forked_children_lock, "release-request", release_lock_attempted
    )

    def _spawn(*args, **kwargs):
        spawn_entered.set()
        assert allow_spawn.wait(timeout=5)
        return _make_mock_process(pid=45)

    def _fork_request():
        with app.test_client() as thread_client:
            response = thread_client.post(
                "/fork",
                json={
                    "role": "worker",
                    "worker_index": 11,
                    "raw_cmd": ["python"],
                },
            )
            statuses["fork"] = response.status_code

    def _release_request():
        with app.test_client() as thread_client:
            response = thread_client.post(
                "/release_ports", json={"role": "worker", "worker_index": 11}
            )
            statuses["release"] = response.status_code

    mock_run.side_effect = _spawn
    fork_thread = threading.Thread(target=_fork_request)
    release_thread = threading.Thread(target=_release_request, name="release-request")

    fork_thread.start()
    assert spawn_entered.wait(timeout=5)
    release_thread.start()
    assert release_lock_attempted.wait(timeout=5)
    allow_spawn.set()
    fork_thread.join(timeout=5)
    release_thread.join(timeout=5)

    assert not fork_thread.is_alive()
    assert not release_thread.is_alive()
    assert statuses == {"fork": 200, "release": 409}
    assert state.owned_ports[("worker", 11)] == {9021}
    assert state.allocated_ports == {9021}


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


@patch("areal.infra.rpc.guard.app.kill_process_tree")
def test_kill_does_not_remove_replacement_owner(mock_kill, client, state: GuardState):
    """A stale concurrent kill cannot clean up a newer child generation."""
    key = ("test", 2)
    old_process = _make_mock_process(pid=125)
    replacement_process = _make_mock_process(pid=126)
    state.forked_children.append(old_process)
    state.forked_children_map[key] = old_process
    state.owned_ports[key] = {9022}
    state.allocated_ports.add(9022)

    def _replace_owner(*args, **kwargs):
        with state.forked_children_lock, state.allocated_ports_lock:
            state.forked_children.remove(old_process)
            state.forked_children.append(replacement_process)
            state.forked_children_map[key] = replacement_process
            state.allocated_ports.remove(9022)
            state.allocated_ports.add(9023)
            state.owned_ports[key] = {9023}

    mock_kill.side_effect = _replace_owner

    response = client.post(
        "/kill_forked_worker", json={"role": "test", "worker_index": 2}
    )

    assert response.status_code == 200
    assert response.get_json()["released_ports"] == []
    assert state.forked_children_map[key] is replacement_process
    assert state.owned_ports[key] == {9023}
    assert state.allocated_ports == {9023}


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
