# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import aiohttp
import pytest

from areal.api import Worker
from areal.api.cli_args import NameResolveConfig
from areal.infra.rpc.guard import app as guard_app
from areal.infra.rpc.guard.app import (
    GuardState,
    begin_fork_request_drain,
    cleanup_forked_children,
    cleanup_port_reservations,
    configure_state_from_args,
    create_app,
    wait_for_fork_request_drain,
)
from areal.infra.scheduler.exceptions import WorkerCleanupError, WorkerCreationError
from areal.infra.scheduler.fork_utils import ForkOwnership, ForkRoleState
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo
from areal.infra.scheduler.ray import RayScheduler, RayWorkerInfo
from areal.infra.scheduler.slurm import SlurmScheduler, SlurmWorkerInfo
from areal.infra.utils.launcher import JobState
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports


def _worker(role: str, port: int) -> Worker:
    return Worker(
        id=f"{role}/0",
        ip="127.0.0.1",
        worker_ports=[str(port)],
        engine_ports=[],
    )


def _worker_infos(scheduler_type: type[Any]):
    owner_worker = _worker("actor", 19000)
    child_worker = _worker("teacher", 19001)
    if scheduler_type is LocalScheduler:
        owner = WorkerInfo(
            worker=owner_worker,
            process=None,
            role="actor",
            gpu_devices=[0],
            created_at=0.0,
            log_file="actor.log",
        )
        child = WorkerInfo(
            worker=child_worker,
            process=None,
            role="teacher",
            gpu_devices=[0],
            created_at=0.0,
            log_file="teacher.log",
        )
    elif scheduler_type is SlurmScheduler:
        owner = SlurmWorkerInfo(
            worker=owner_worker,
            role="actor",
            slurm_job_id=1,
            task_index=0,
        )
        child = SlurmWorkerInfo(
            worker=child_worker,
            role="teacher",
            slurm_job_id=-1,
            task_index=0,
        )
    else:
        owner = RayWorkerInfo(worker=owner_worker, role="actor", task_index=0)
        child = RayWorkerInfo(worker=child_worker, role="teacher", task_index=0)
    return owner, child


def _scheduler(scheduler_type: type[Any]):
    scheduler = object.__new__(scheduler_type)
    owner, child = _worker_infos(scheduler_type)
    scheduler._workers = {"actor": [owner], "teacher": [child]}
    scheduler._colocated_roles = {"teacher": "rollout", "rollout": "actor"}
    scheduler._fork_parent_roles = {"teacher": "actor"}
    if scheduler_type is LocalScheduler:
        scheduler._cleanup_workers = lambda workers: None
    return scheduler


def _fork_test_scheduler(scheduler_type: type[Any]):
    scheduler = object.__new__(scheduler_type)
    scheduler.experiment_name = "test"
    scheduler.trial_name = "response-loss"
    scheduler.fileroot = None
    scheduler.name_resolve_config = NameResolveConfig(type=None)
    if scheduler_type is LocalScheduler:
        scheduler.log_dir = Path("/tmp")
        owner = WorkerInfo(
            worker=_worker("actor", 19000),
            process=None,
            role="actor",
            gpu_devices=[0],
            created_at=0.0,
            log_file="actor.log",
        )
    elif scheduler_type is RayScheduler:
        owner = RayWorkerInfo(
            worker=_worker("actor", 19000),
            role="actor",
            task_index=0,
        )
    else:
        owner = SlurmWorkerInfo(
            worker=_worker("actor", 19000),
            role="actor",
            slurm_job_id=1,
            task_index=0,
        )
    scheduler._workers = {"actor": [owner]}
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    return scheduler, owner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, SlurmScheduler, RayScheduler]
)
async def test_fork_child_receives_role_environment_overrides(
    monkeypatch, scheduler_type
):
    """Forked roles override the owner environment before importing CUDA users."""

    class FakeResponse:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self, content_type=None):
            return self.payload

        async def text(self):
            return str(self.payload)

    class FakeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json):
            self.calls.append((url, json))
            if url.endswith("/reserve_worker_ports"):
                return FakeResponse({"host": "127.0.0.1", "ports": [20001]})
            if url.endswith("/fork"):
                return FakeResponse({"status": "success", "pid": 1234})
            raise AssertionError(f"Unexpected URL: {url}")

    scheduler, owner = _fork_test_scheduler(scheduler_type)

    async def ready(*args, **kwargs):
        return True

    monkeypatch.setattr(scheduler, "_wait_for_fork_ready", ready)
    session = FakeSession()
    env = {"PYTORCH_CUDA_ALLOC_CONF": ""}

    worker = await scheduler._fork_single_worker(
        session,
        "rollout",
        0,
        owner,
        "actor",
        env=env,
    )

    fork_payload = next(
        payload for url, payload in session.calls if url.endswith("/fork")
    )
    assert fork_payload["env"] == env
    if scheduler_type is LocalScheduler:
        assert worker.env_vars["PYTORCH_CUDA_ALLOC_CONF"] == ""


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, SlurmScheduler, RayScheduler]
)
def test_delete_fork_role_kills_actual_owner_before_metadata(
    scheduler_type: type[Any],
):
    """Every scheduler kills a fork child through its resolved actor guard."""
    scheduler = _scheduler(scheduler_type)
    calls = []

    async def cleanup(role, target_role, workers):
        calls.append((role, target_role, list(workers)))
        assert role in scheduler._workers
        assert role in scheduler._colocated_roles

    scheduler._cleanup_forked_workers_async = cleanup

    scheduler.delete_workers("teacher")

    assert calls[0][0:2] == ("teacher", "actor")
    assert "teacher" not in scheduler._workers
    assert "teacher" not in scheduler._colocated_roles
    assert "actor" in scheduler._workers


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, SlurmScheduler, RayScheduler]
)
def test_delete_fork_role_failure_preserves_retry_metadata(
    scheduler_type: type[Any],
):
    """A failed guard kill is surfaced and keeps child ownership retryable."""
    scheduler = _scheduler(scheduler_type)

    async def cleanup(role, target_role, workers):
        raise WorkerCleanupError(role, ["rank 0 timed out"])

    scheduler._cleanup_forked_workers_async = cleanup

    with pytest.raises(WorkerCleanupError, match="rank 0 timed out"):
        scheduler.delete_workers("teacher")

    assert "teacher" in scheduler._workers
    assert scheduler._colocated_roles["teacher"] == "rollout"
    assert scheduler._fork_parent_roles["teacher"] == "actor"


def test_alias_owner_and_leaf_first_order_resolve_nested_roles():
    """Nested rollout aliases resolve to actor and delete leaves before parents."""
    scheduler = _scheduler(LocalScheduler)
    scheduler._colocated_roles.update(
        {
            "eval-rollout": "rollout",
            "proxy-rollout": "rollout",
            "proxy-eval-rollout": "eval-rollout",
        }
    )

    owner = scheduler._resolve_worker_owner("eval-rollout")
    order = scheduler._colocated_roles_leaf_first()

    assert owner == "actor"
    assert order.index("proxy-eval-rollout") < order.index("eval-rollout")
    assert order.index("eval-rollout") < order.index("rollout")
    assert order.index("proxy-rollout") < order.index("rollout")


def test_alias_owner_cycle_raises():
    """Invalid alias cycles fail explicitly instead of looping forever."""
    scheduler = object.__new__(LocalScheduler)
    scheduler._workers = {}
    scheduler._colocated_roles = {"rollout": "eval-rollout", "eval-rollout": "rollout"}

    with pytest.raises(ValueError, match="cycle"):
        scheduler._resolve_worker_owner("rollout")


@pytest.mark.parametrize("scheduler_type", [LocalScheduler, RayScheduler])
def test_delete_owner_removes_colocated_descendants_before_processes(scheduler_type):
    """Role-specific owner deletion is leaf-first, not only delete-all."""
    scheduler = _scheduler(scheduler_type)
    events = []

    async def cleanup(role, target_role, workers):
        events.append(("fork", role))

    scheduler._cleanup_forked_workers_async = cleanup
    if scheduler_type is LocalScheduler:
        scheduler._allocated_ports = set()
        scheduler._cleanup_workers = lambda workers: events.append(
            ("ports", workers[0].worker.id.split("/")[0])
        )
    else:
        scheduler._launchers = {"actor": []}
        scheduler._placement_groups = {}
        scheduler._destroy_engines_on_workers = lambda workers: events.append(
            ("process", "actor")
        )
        scheduler._stop_launchers = lambda role, timeout: []

    scheduler.delete_workers("actor")

    if scheduler_type is LocalScheduler:
        assert events == [("fork", "teacher"), ("ports", "teacher"), ("ports", "actor")]
    else:
        assert events == [("fork", "teacher"), ("process", "actor")]
    assert scheduler._workers == {}
    assert scheduler._colocated_roles == {}


@pytest.mark.parametrize("scheduler_type", [LocalScheduler, RayScheduler])
def test_delete_owner_stops_when_descendant_cleanup_fails(scheduler_type):
    """A failed child cleanup retains the process owner for retry."""
    scheduler = _scheduler(scheduler_type)

    async def cleanup(role, target_role, workers):
        raise WorkerCleanupError(role, ["guard unavailable"])

    scheduler._cleanup_forked_workers_async = cleanup

    with pytest.raises(WorkerCleanupError, match="guard unavailable"):
        scheduler.delete_workers("actor")

    assert "actor" in scheduler._workers
    assert "teacher" in scheduler._workers


def test_local_delete_failure_retains_role_and_ports_for_retry(monkeypatch):
    """A failed process kill cannot orphan role metadata or free its ports."""
    scheduler = object.__new__(LocalScheduler)
    worker = WorkerInfo(
        worker=_worker("actor", 19000),
        process=Namespace(pid=12345),
        role="actor",
        gpu_devices=[0],
        created_at=0.0,
        log_file="actor.log",
    )
    scheduler._workers = {"actor": [worker]}
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    scheduler._allocated_ports = {19000}

    def fail_kill(*args, **kwargs):
        raise RuntimeError("kill failed")

    monkeypatch.setattr("areal.infra.scheduler.local.kill_process_tree", fail_kill)

    with pytest.raises(WorkerCleanupError, match="kill failed"):
        scheduler.delete_workers("actor")

    assert "actor" in scheduler._workers
    assert scheduler._allocated_ports == {19000}
    monkeypatch.setattr(
        "areal.infra.scheduler.local.kill_process_tree",
        lambda *args, **kwargs: None,
    )
    scheduler.delete_workers("actor")
    assert scheduler._workers == {}
    assert scheduler._allocated_ports == set()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_guard_kill_forked_worker_reaps_pid_releases_port_and_is_idempotent(
    tmp_path,
):
    """The guard owns child lifetime and releases its reserved port after exit."""
    state = GuardState()
    state.server_host = "127.0.0.1"
    state.fileroot = str(tmp_path)
    state.experiment_name = "mopd-test"
    state.trial_name = "trial"
    client = create_app(state).test_client()

    port_response = client.post("/alloc_ports", json={"count": 1})
    port = port_response.get_json()["ports"][0]
    fork_response = client.post(
        "/fork",
        json={
            "role": "teacher",
            "worker_index": 0,
            "raw_cmd": [sys.executable, "-c", "import time; time.sleep(60)"],
            "allocated_ports": [port],
        },
    )
    pid = fork_response.get_json()["pid"]

    try:
        assert fork_response.status_code == 200
        assert _pid_exists(pid)
        assert port in state.allocated_ports

        kill_response = client.post(
            "/kill_forked_worker",
            json={"role": "teacher", "worker_index": 0},
        )
        deadline = time.monotonic() + 5
        while _pid_exists(pid) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert kill_response.status_code == 200
        assert not _pid_exists(pid)
        assert port not in state.allocated_ports
        assert state.forked_children_map == {}

        repeated_response = client.post(
            "/kill_forked_worker",
            json={"role": "teacher", "worker_index": 0},
        )
        assert repeated_response.status_code == 200
    finally:
        if _pid_exists(pid):
            kill_process_tree(pid, timeout=1, graceful=False)
        cleanup_port_reservations(state)


def test_guard_alloc_ports_cross_process_candidate_collision_uses_new_port(
    monkeypatch,
):
    """Independent Guards use a node-wide lease before returning a port."""
    first_port, second_port = find_free_ports(2)
    candidates = iter(([first_port], [first_port], [second_port]))

    def fake_find_free_ports(count, exclude_ports):
        assert count == 1
        return next(candidates)

    monkeypatch.setattr(guard_app, "find_free_ports", fake_find_free_ports)
    first_state = GuardState()
    second_state = GuardState()
    first_client = create_app(first_state).test_client()
    second_client = create_app(second_state).test_client()

    try:
        first_response = first_client.post("/alloc_ports", json={"count": 1})
        second_response = second_client.post("/alloc_ports", json={"count": 1})

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.get_json()["ports"] == [first_port]
        assert second_response.get_json()["ports"] == [second_port]
        assert set(first_state.port_reservations) == {first_port}
        assert set(second_state.port_reservations) == {second_port}
    finally:
        cleanup_port_reservations(first_state)
        cleanup_port_reservations(second_state)


def test_guard_fixed_worker_ports_survive_child_restart(tmp_path):
    """A repeated fork reuses its fixed port group for the experiment lifetime."""
    state = GuardState()
    state.server_host = "127.0.0.1"
    state.fileroot = str(tmp_path)
    state.experiment_name = "fixed-port-test"
    state.trial_name = "trial"
    client = create_app(state).test_client()
    reserve_payload = {
        "role": "mopd-teacher",
        "worker_index": 0,
        "count": 3,
    }

    first_reservation = client.post(
        "/reserve_worker_ports",
        json=reserve_payload,
    )
    ports = first_reservation.get_json()["ports"]
    fork_response = client.post(
        "/fork",
        json={
            "role": "mopd-teacher",
            "worker_index": 0,
            "raw_cmd": [sys.executable, "-c", "import time; time.sleep(60)"],
            "allocated_ports": ports,
        },
    )
    pid = fork_response.get_json()["pid"]
    repeated_fork = client.post(
        "/fork",
        json={
            "role": "mopd-teacher",
            "worker_index": 0,
            "raw_cmd": [sys.executable, "-c", "import time; time.sleep(60)"],
            "allocated_ports": ports,
        },
    )

    try:
        assert first_reservation.status_code == 200
        assert len(ports) == 3
        assert fork_response.status_code == 200
        assert repeated_fork.status_code == 200
        assert repeated_fork.get_json()["pid"] == pid
        assert repeated_fork.get_json()["reused"] is True

        kill_response = client.post(
            "/kill_forked_worker",
            json={"role": "mopd-teacher", "worker_index": 0},
        )
        second_reservation = client.post(
            "/reserve_worker_ports",
            json=reserve_payload,
        )

        assert kill_response.status_code == 200
        assert second_reservation.status_code == 200
        assert second_reservation.get_json()["ports"] == ports
        assert set(ports).issubset(state.allocated_ports)
        assert state.fixed_worker_ports[("mopd-teacher", 0)] == tuple(ports)

        formal_delete = client.post(
            "/kill_forked_worker",
            json={
                "role": "mopd-teacher",
                "worker_index": 0,
                "release_ports": True,
            },
        )
        repeated_delete = client.post(
            "/kill_forked_worker",
            json={
                "role": "mopd-teacher",
                "worker_index": 0,
                "release_ports": True,
            },
        )
        assert formal_delete.status_code == 200
        assert repeated_delete.status_code == 200
        assert ("mopd-teacher", 0) not in state.fixed_worker_ports
        assert set(ports).isdisjoint(state.allocated_ports)
    finally:
        if _pid_exists(pid):
            kill_process_tree(pid, timeout=1, graceful=False)
        cleanup_port_reservations(state)


def test_guard_same_key_kill_and_restart_are_linearized(monkeypatch, tmp_path):
    """A blocked kill cannot remove a concurrently restarted child or its lease."""

    class FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return self.returncode

    state = GuardState()
    state.server_host = "127.0.0.1"
    state.fileroot = str(tmp_path)
    state.experiment_name = "lifecycle-race"
    state.trial_name = "trial"
    old_process = FakeProcess(1001)
    new_process = FakeProcess(1002)
    key = ("teacher", 0)
    initial_client = create_app(state).test_client()
    initial_reservation = initial_client.post(
        "/reserve_worker_ports",
        json={"role": key[0], "worker_index": key[1], "count": 1},
    ).get_json()["ports"]
    state.forked_children.append(old_process)
    state.forked_children_map[key] = old_process
    state.forked_children_ports[key] = set(initial_reservation)

    kill_started = Event()
    allow_kill = Event()
    spawn_called = Event()

    def blocked_kill(pid, timeout, graceful):
        assert pid == old_process.pid
        kill_started.set()
        assert allow_kill.wait(timeout=5)
        old_process.returncode = 0

    def fake_spawn(*args, **kwargs):
        spawn_called.set()
        return new_process

    monkeypatch.setattr(guard_app, "kill_process_tree", blocked_kill)
    monkeypatch.setattr(guard_app, "run_with_streaming_logs", fake_spawn)

    def kill_old():
        return (
            create_app(state)
            .test_client()
            .post(
                "/kill_forked_worker",
                json={"role": key[0], "worker_index": key[1], "release_ports": True},
            )
        )

    def reserve_and_fork_new():
        client = create_app(state).test_client()
        reservation = client.post(
            "/reserve_worker_ports",
            json={"role": key[0], "worker_index": key[1], "count": 1},
        )
        ports = reservation.get_json()["ports"]
        fork = client.post(
            "/fork",
            json={
                "role": key[0],
                "worker_index": key[1],
                "raw_cmd": [sys.executable, "-c", "pass"],
                "allocated_ports": ports,
            },
        )
        return reservation, fork, ports

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            kill_future = executor.submit(kill_old)
            assert kill_started.wait(timeout=5)
            restart_future = executor.submit(reserve_and_fork_new)
            assert not spawn_called.wait(timeout=0.1)
            allow_kill.set()
            kill_response = kill_future.result(timeout=5)
            reserve_response, fork_response, new_ports = restart_future.result(
                timeout=5
            )

        assert kill_response.status_code == 200
        assert reserve_response.status_code == 200
        assert fork_response.status_code == 200
        assert state.forked_children_map[key] is new_process
        assert state.forked_children_ports[key] == set(new_ports)
        assert state.fixed_worker_ports[key] == tuple(new_ports)
        assert set(new_ports).issubset(state.allocated_ports)
    finally:
        cleanup_port_reservations(state)


def test_guard_reservation_only_delete_and_reserve_are_linearized(monkeypatch):
    """Deleting a reservation-only key cannot pop a concurrent new lease."""
    state = GuardState()
    state.server_host = "127.0.0.1"
    key = ("teacher", 0)
    initial_client = create_app(state).test_client()
    initial_client.post(
        "/reserve_worker_ports",
        json={"role": key[0], "worker_index": key[1], "count": 1},
    )
    release_started = Event()
    allow_release = Event()
    original_release = guard_app._release_reserved_ports_unlocked

    def blocked_release(guard_state, ports):
        release_started.set()
        assert allow_release.wait(timeout=5)
        original_release(guard_state, ports)

    monkeypatch.setattr(guard_app, "_release_reserved_ports_unlocked", blocked_release)

    def delete_reservation():
        return (
            create_app(state)
            .test_client()
            .post(
                "/kill_forked_worker",
                json={"role": key[0], "worker_index": key[1], "release_ports": True},
            )
        )

    def reserve_again():
        return (
            create_app(state)
            .test_client()
            .post(
                "/reserve_worker_ports",
                json={"role": key[0], "worker_index": key[1], "count": 1},
            )
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            delete_future = executor.submit(delete_reservation)
            assert release_started.wait(timeout=5)
            reserve_future = executor.submit(reserve_again)
            time.sleep(0.1)
            assert not reserve_future.done()
            allow_release.set()
            delete_response = delete_future.result(timeout=5)
            reserve_response = reserve_future.result(timeout=5)

        new_ports = reserve_response.get_json()["ports"]
        assert delete_response.status_code == 200
        assert reserve_response.status_code == 200
        assert state.fixed_worker_ports[key] == tuple(new_ports)
        assert set(new_ports).issubset(state.allocated_ports)
    finally:
        allow_release.set()
        cleanup_port_reservations(state)


def test_guard_rejects_fork_after_formal_port_release(monkeypatch):
    """A fork queued behind formal deletion cannot spawn with a stale lease."""
    state = GuardState()
    state.server_host = "127.0.0.1"
    client = create_app(state).test_client()
    key = ("teacher", 0)
    ports = client.post(
        "/reserve_worker_ports",
        json={"role": key[0], "worker_index": key[1], "count": 1},
    ).get_json()["ports"]
    delete_response = client.post(
        "/kill_forked_worker",
        json={"role": key[0], "worker_index": key[1], "release_ports": True},
    )
    monkeypatch.setattr(
        guard_app,
        "run_with_streaming_logs",
        lambda *args, **kwargs: pytest.fail("stale fork must not spawn"),
    )

    try:
        fork_response = client.post(
            "/fork",
            json={
                "role": key[0],
                "worker_index": key[1],
                "raw_cmd": [sys.executable, "-c", "pass"],
                "allocated_ports": ports,
            },
        )

        assert delete_response.status_code == 200
        assert fork_response.status_code == 409
        assert "stale or unreserved" in fork_response.get_json()["error"]
        assert key not in state.forked_children_map
    finally:
        cleanup_port_reservations(state)


def test_guard_shutdown_drains_admitted_fork_before_cleanup(monkeypatch, tmp_path):
    """Shutdown rejects new work and reaps a child from an in-flight fork."""

    class FakeProcess:
        pid = 1003
        returncode = None

        def poll(self):
            return self.returncode

    state = GuardState()
    state.server_host = "127.0.0.1"
    state.fileroot = str(tmp_path)
    state.experiment_name = "shutdown-race"
    state.trial_name = "trial"
    client = create_app(state).test_client()
    ports = client.post(
        "/reserve_worker_ports",
        json={"role": "teacher", "worker_index": 0, "count": 1},
    ).get_json()["ports"]
    process = FakeProcess()
    spawn_started = Event()
    allow_spawn = Event()
    killed = Event()

    def blocked_spawn(*args, **kwargs):
        spawn_started.set()
        assert allow_spawn.wait(timeout=5)
        return process

    def stop_process(*args, **kwargs):
        process.returncode = 0
        killed.set()

    monkeypatch.setattr(guard_app, "run_with_streaming_logs", blocked_spawn)
    monkeypatch.setattr(guard_app, "kill_process_tree", stop_process)

    def fork_worker():
        return (
            create_app(state)
            .test_client()
            .post(
                "/fork",
                json={
                    "role": "teacher",
                    "worker_index": 0,
                    "raw_cmd": [sys.executable, "-c", "pass"],
                    "allocated_ports": ports,
                },
            )
        )

    def drain_requests():
        begin_fork_request_drain(state)
        wait_for_fork_request_drain(state)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fork_future = executor.submit(fork_worker)
            assert spawn_started.wait(timeout=5)
            drain_future = executor.submit(drain_requests)
            time.sleep(0.1)
            assert not drain_future.done()

            rejected = (
                create_app(state)
                .test_client()
                .post(
                    "/reserve_worker_ports",
                    json={"role": "late", "worker_index": 0, "count": 1},
                )
            )
            assert rejected.status_code == 503
            assert ("late", 0) not in state.fixed_worker_ports

            allow_spawn.set()
            assert fork_future.result(timeout=5).status_code == 200
            drain_future.result(timeout=5)

        cleanup_forked_children(state)
        cleanup_port_reservations(state)
        assert killed.is_set()
        assert state.forked_children == []
        assert state.forked_children_map == {}
        assert state.fixed_worker_ports == {}
        assert state.allocated_ports == set()
    finally:
        allow_spawn.set()
        cleanup_port_reservations(state)


def test_guard_shutdown_retains_failed_child_tracking_and_port_lease(monkeypatch):
    """Shutdown releases only ports whose child is confirmed stopped."""

    class FakeProcess:
        def __init__(self, pid: int):
            self.pid = pid
            self.returncode = None

        def poll(self):
            return self.returncode

    state = GuardState()
    successful = FakeProcess(1004)
    failed = FakeProcess(1005)
    successful_key = ("teacher", 0)
    failed_key = ("teacher", 1)
    with state.allocated_ports_lock:
        successful_port, failed_port = guard_app._reserve_free_ports(state, 2)
    successful_lease = state.port_reservations[successful_port]
    failed_lease = state.port_reservations[failed_port]
    state.forked_children = [successful, failed]
    state.forked_children_map = {
        successful_key: successful,
        failed_key: failed,
    }
    state.forked_children_ports = {
        successful_key: {successful_port},
        failed_key: {failed_port},
    }
    state.fixed_worker_ports = {
        successful_key: (successful_port,),
        failed_key: (failed_port,),
    }

    def stop_process(pid, timeout, graceful):
        del timeout, graceful
        if pid == successful.pid:
            assert successful_port in state.port_reservations
            successful.returncode = 0
            return
        assert failed_port in state.port_reservations
        raise RuntimeError("termination failed")

    monkeypatch.setattr(guard_app, "kill_process_tree", stop_process)

    try:
        preserved_ports = cleanup_forked_children(state)
        cleanup_port_reservations(state, preserve_ports=preserved_ports)

        assert preserved_ports == {failed_port}
        assert successful_key not in state.forked_children_map
        assert successful_port not in state.port_reservations
        assert successful_lease.fileno() == -1
        assert state.forked_children == [failed]
        assert state.forked_children_map[failed_key] is failed
        assert state.forked_children_ports[failed_key] == {failed_port}
        assert state.fixed_worker_ports[failed_key] == (failed_port,)
        assert failed_port in state.allocated_ports
        assert state.port_reservations[failed_port] is failed_lease
        assert failed_lease.fileno() >= 0

        failed.returncode = 0
        assert cleanup_forked_children(state) == set()
        assert failed_key not in state.forked_children_map
        assert failed_port not in state.port_reservations
        assert failed_lease.fileno() == -1
    finally:
        cleanup_port_reservations(state)


def test_guard_shutdown_preserves_lease_when_child_remains_alive(monkeypatch):
    """A kill call that returns without stopping the child retains ownership."""

    class FakeProcess:
        pid = 1006

        def poll(self):
            return None

    state = GuardState()
    process = FakeProcess()
    key = ("teacher", 0)
    with state.allocated_ports_lock:
        [port] = guard_app._reserve_free_ports(state, 1)
    state.forked_children = [process]
    state.forked_children_map[key] = process
    state.forked_children_ports[key] = {port}
    state.fixed_worker_ports[key] = (port,)
    monkeypatch.setattr(guard_app, "kill_process_tree", lambda *args, **kwargs: None)

    try:
        preserved_ports = cleanup_forked_children(state)

        assert preserved_ports == {port}
        assert state.forked_children == [process]
        assert state.forked_children_map[key] is process
        assert port in state.port_reservations
    finally:
        state.forked_children.clear()
        state.forked_children_map.clear()
        state.forked_children_ports.clear()
        cleanup_port_reservations(state)


def test_guard_shutdown_closes_listener_before_final_cleanup(monkeypatch):
    """The shutdown sequence drains the listener before hooks and ownership."""
    state = GuardState()
    events = []
    state.register_cleanup_hook(lambda: events.append("hook"))
    server = Namespace(
        shutdown=lambda: events.append("shutdown"),
        server_close=lambda: events.append("server_close"),
    )
    monkeypatch.setattr(
        guard_app,
        "begin_fork_request_drain",
        lambda _state: events.append("begin"),
    )
    monkeypatch.setattr(
        guard_app,
        "wait_for_fork_request_drain",
        lambda _state: events.append("drain"),
    )
    monkeypatch.setattr(
        guard_app,
        "cleanup_forked_children",
        lambda _state: events.append("children") or set(),
    )
    monkeypatch.setattr(
        guard_app,
        "cleanup_port_reservations",
        lambda _state, preserve_ports=None: events.append("ports"),
    )

    guard_app._shutdown_guard(state, server)

    assert events == [
        "begin",
        "shutdown",
        "server_close",
        "drain",
        "hook",
        "children",
        "ports",
    ]


def test_guard_health_identifies_exact_worker():
    """Health responses identify the process behind a forked endpoint."""
    state = GuardState()
    state.role = "mopd-teacher"
    state.worker_index = 33

    response = create_app(state).test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json()["role"] == "mopd-teacher"
    assert response.get_json()["worker_index"] == 33
    assert response.get_json()["pid"] == os.getpid()
    assert response.get_json()["generation"] == state.generation


def test_guard_explicit_worker_index_wins_over_stale_slurm_env(monkeypatch):
    """Local workers keep their CLI identity inside a Slurm login shell."""
    monkeypatch.setenv("SLURM_PROCID", "0")
    state = GuardState()
    args = Namespace(
        host="127.0.0.1",
        experiment_name="test",
        trial_name="local",
        role="actor",
        worker_index=7,
        name_resolve_type=None,
        nfs_record_root=None,
        etcd3_addr=None,
        fileroot=None,
    )

    configure_state_from_args(state, args)

    assert state.worker_index == 7


def test_guard_uses_slurm_worker_index_when_cli_index_is_missing(monkeypatch):
    """Slurm launchers can still supply task identity through the environment."""
    monkeypatch.setenv("SLURM_PROCID", "5")
    state = GuardState()
    args = Namespace(
        host="127.0.0.1",
        experiment_name="test",
        trial_name="slurm",
        role="actor",
        worker_index=-1,
        name_resolve_type=None,
        nfs_record_root=None,
        etcd3_addr=None,
        fileroot=None,
    )

    configure_state_from_args(state, args)

    assert state.worker_index == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_type", [LocalScheduler, RayScheduler])
@pytest.mark.parametrize("guard_state", ["alive", "missing", "unknown"])
async def test_fork_response_loss_reconciles_or_retains_provisional_ownership(
    monkeypatch,
    scheduler_type,
    guard_state,
):
    """A lost fork response cannot create an unowned child or port lease."""

    class FakeResponse:
        def __init__(self, payload, status=200, enter_error=None):
            self.payload = payload
            self.status = status
            self.enter_error = enter_error

        async def __aenter__(self):
            if self.enter_error is not None:
                raise self.enter_error
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self, content_type=None):
            return self.payload

        async def text(self):
            return str(self.payload)

    class FakeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json):
            self.calls.append((url, json))
            if url.endswith("/reserve_worker_ports"):
                return FakeResponse({"host": "127.0.0.1", "ports": [20001]})
            if url.endswith("/fork"):
                return FakeResponse(
                    {}, enter_error=aiohttp.ClientConnectionError("response lost")
                )
            if url.endswith("/forked_worker_status"):
                if guard_state == "unknown":
                    return FakeResponse({"error": "unavailable"}, status=503)
                return FakeResponse(
                    {
                        "status": "success",
                        "alive": guard_state == "alive",
                        "pid": 1234 if guard_state == "alive" else None,
                        "ports": [20001] if guard_state == "alive" else [],
                    }
                )
            if url.endswith("/kill_forked_worker"):
                return FakeResponse({"status": "success"})
            raise AssertionError(f"Unexpected URL: {url}")

    scheduler, owner = _fork_test_scheduler(scheduler_type)

    async def ready(*args, **kwargs):
        return True

    monkeypatch.setattr(scheduler, "_wait_for_fork_ready", ready)
    session = FakeSession()

    if guard_state == "alive":
        worker = await scheduler._fork_single_worker(
            session, "teacher", 0, owner, "actor"
        )
        assert worker.worker.worker_ports == ["20001"]
        assert scheduler._workers["teacher"] == [worker]
    elif guard_state == "missing":
        with pytest.raises(WorkerCreationError, match="response lost"):
            await scheduler._fork_single_worker(session, "teacher", 0, owner, "actor")
        assert "teacher" not in scheduler._workers
        assert any(url.endswith("/kill_forked_worker") for url, _ in session.calls)
    else:
        with pytest.raises(WorkerCleanupError, match="uncertain fork outcome"):
            await scheduler._fork_single_worker(session, "teacher", 0, owner, "actor")
        assert scheduler._workers["teacher"][0].worker.worker_ports == ["20001"]
        assert scheduler._fork_parent_roles["teacher"] == "actor"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
async def test_reserve_response_loss_retries_same_key(monkeypatch, scheduler_type):
    """A lost reserve response is recovered by the Guard's idempotent key."""

    class FakeResponse:
        status = 200

        def __init__(self, payload=None, enter_error=None):
            self.payload = payload
            self.enter_error = enter_error

        async def __aenter__(self):
            if self.enter_error is not None:
                raise self.enter_error
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self, content_type=None):
            return self.payload

        async def text(self):
            return str(self.payload)

    class FakeSession:
        def __init__(self):
            self.calls = []
            self.reserve_calls = 0

        def post(self, url, json):
            self.calls.append((url, json))
            if url.endswith("/reserve_worker_ports"):
                self.reserve_calls += 1
                if self.reserve_calls == 1:
                    return FakeResponse(
                        enter_error=aiohttp.ClientConnectionError("response lost")
                    )
                return FakeResponse({"host": "127.0.0.1", "ports": [20001]})
            if url.endswith("/fork"):
                return FakeResponse({"status": "success", "pid": 1234})
            raise AssertionError(f"Unexpected URL: {url}")

    scheduler, owner = _fork_test_scheduler(scheduler_type)

    async def ready(*args, **kwargs):
        return True

    monkeypatch.setattr(scheduler, "_wait_for_fork_ready", ready)
    session = FakeSession()

    worker = await scheduler._fork_single_worker(session, "teacher", 0, owner, "actor")

    reserve_payloads = [
        payload
        for url, payload in session.calls
        if url.endswith("/reserve_worker_ports")
    ]
    assert len(reserve_payloads) == 2
    assert reserve_payloads[0] == reserve_payloads[1]
    assert worker.worker.worker_ports == ["20001"]
    assert scheduler._fork_reservations["teacher"].reservation_indices == {0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
async def test_repeated_reserve_loss_retains_public_cleanup_ownership(
    scheduler_type,
):
    """Repeated reserve loss leaves a key that public deletion can release."""

    class LostResponse:
        async def __aenter__(self):
            raise aiohttp.ClientConnectionError("guard unreachable")

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, json):
            self.calls.append((url, json))
            return LostResponse()

    scheduler, owner = _fork_test_scheduler(scheduler_type)
    session = FakeSession()

    with pytest.raises(WorkerCreationError, match="guard unreachable"):
        await scheduler._fork_single_worker(session, "teacher", 0, owner, "actor")

    assert len(session.calls) == 2
    assert scheduler._fork_reservations["teacher"].reservation_indices == {0}
    assert (
        scheduler._fork_reservations["teacher"].state is ForkRoleState.CLEANUP_PENDING
    )
    assert scheduler._fork_parent_roles["teacher"] == "actor"
    assert "teacher" not in scheduler._workers
    cleanup_calls = []

    async def kill_reservation(session, role, idx, target_wi):
        cleanup_calls.append((role, idx, target_wi.worker.id))

    scheduler._kill_forked_worker = kill_reservation
    if scheduler_type is LocalScheduler:
        scheduler._cleanup_workers = lambda workers: None

    scheduler.delete_workers("teacher")

    assert cleanup_calls == [("teacher", 0, "actor/0")]
    assert "teacher" not in scheduler._fork_reservations
    assert "teacher" not in scheduler._colocated_roles


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
def test_public_fork_failure_preserves_provisional_ownership(
    scheduler_type,
):
    """Public fork failure keeps retryable state until delete_workers succeeds."""
    scheduler = _scheduler(scheduler_type)
    scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    scheduler._fork_reservations = {}
    _, child = _worker_infos(scheduler_type)

    async def fail_after_retaining(role, target_role, workers, command, env_vars=None):
        scheduler._retain_fork_workers(role, target_role, [child])
        scheduler._fork_reservations[role] = ForkOwnership(
            owner_role=target_role,
            reservation_indices={0},
        )
        raise WorkerCleanupError(role, ["rollback unavailable"])

    scheduler._create_forked_workers_async = fail_after_retaining

    with pytest.raises(WorkerCleanupError, match="rollback unavailable"):
        scheduler.fork_workers("teacher", "actor")

    assert scheduler._workers["teacher"] == [child]
    assert scheduler._colocated_roles["teacher"] == "actor"
    assert scheduler._fork_parent_roles["teacher"] == "actor"
    assert scheduler._fork_reservations["teacher"].reservation_indices == {0}
    assert (
        scheduler._fork_reservations["teacher"].state is ForkRoleState.CLEANUP_PENDING
    )

    async def cleanup(role, target_role, workers):
        assert (role, target_role, workers) == ("teacher", "actor", [child])

    scheduler._cleanup_forked_workers_async = cleanup
    if scheduler_type is LocalScheduler:
        scheduler._cleanup_workers = lambda workers: None
    scheduler.delete_workers("teacher")

    assert "teacher" not in scheduler._workers
    assert "teacher" not in scheduler._colocated_roles
    assert "teacher" not in scheduler._fork_reservations


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
@pytest.mark.parametrize("with_provisional_worker", [False, True])
def test_cleanup_pending_role_rejects_query_and_recreation(
    scheduler_type,
    with_provisional_worker,
):
    """Cleanup-pending ownership cannot masquerade as a ready alias."""
    scheduler = _scheduler(scheduler_type)
    if not with_provisional_worker:
        scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {"teacher": "actor"}
    scheduler._fork_parent_roles = {"teacher": "actor"}
    scheduler._fork_reservations = {
        "teacher": ForkOwnership(
            owner_role="actor",
            reservation_indices={0},
            state=ForkRoleState.CLEANUP_PENDING,
        )
    }

    with pytest.raises(WorkerCleanupError, match="cleanup_pending"):
        scheduler.get_workers("teacher")
    with pytest.raises(WorkerCreationError, match="still owns resources"):
        scheduler.create_workers(Namespace(role="teacher"))
    with pytest.raises(WorkerCreationError, match="still owns resources"):
        scheduler.fork_workers("teacher", "actor")

    assert scheduler._fork_parent_roles["teacher"] == "actor"
    assert scheduler._fork_reservations["teacher"].reservation_indices == {0}


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
def test_cleanup_pending_role_cannot_be_used_as_fork_target(scheduler_type):
    """Owner resolution rejects a pending role before creating a new child."""
    scheduler = _scheduler(scheduler_type)
    scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {"teacher": "actor"}
    scheduler._fork_parent_roles = {"teacher": "actor"}
    scheduler._fork_reservations = {
        "teacher": ForkOwnership(
            owner_role="actor",
            reservation_indices={0},
            state=ForkRoleState.CLEANUP_PENDING,
        )
    }

    with pytest.raises(WorkerCleanupError, match="cleanup_pending"):
        scheduler.fork_workers("student", "teacher")

    assert "student" not in scheduler._workers
    assert "student" not in scheduler._colocated_roles


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
def test_active_fork_ownership_remains_queryable(scheduler_type):
    """An active fork role still returns its own workers, not its owner."""
    scheduler = _scheduler(scheduler_type)
    scheduler._colocated_roles = {"teacher": "actor"}
    scheduler._fork_reservations = {
        "teacher": ForkOwnership(
            owner_role="actor",
            reservation_indices={0},
            state=ForkRoleState.ACTIVE,
        )
    }
    scheduler._is_worker_ready = lambda worker: True
    if scheduler_type is LocalScheduler:
        scheduler.startup_timeout = 1
        scheduler._check_worker_health = lambda role: None

    workers = scheduler.get_workers("teacher")

    assert [worker.id for worker in workers] == ["teacher/0"]


@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
def test_active_delete_failure_stays_pending_until_retry_succeeds(scheduler_type):
    """A failed active-role delete blocks reuse until a successful retry."""
    scheduler = _scheduler(scheduler_type)
    scheduler._colocated_roles = {"teacher": "actor"}
    scheduler._fork_parent_roles = {"teacher": "actor"}
    scheduler._fork_reservations = {
        "teacher": ForkOwnership(
            owner_role="actor",
            reservation_indices={0},
            state=ForkRoleState.ACTIVE,
        )
    }

    async def fail_cleanup(role, target_role, workers):
        raise WorkerCleanupError(role, ["guard unavailable"])

    scheduler._cleanup_forked_workers_async = fail_cleanup
    with pytest.raises(WorkerCleanupError, match="guard unavailable"):
        scheduler.delete_workers("teacher")

    assert (
        scheduler._fork_reservations["teacher"].state is ForkRoleState.CLEANUP_PENDING
    )
    with pytest.raises(WorkerCleanupError, match="cleanup_pending"):
        scheduler.get_workers("teacher")
    with pytest.raises(WorkerCreationError, match="still owns resources"):
        scheduler.fork_workers("teacher", "actor")

    async def cleanup(role, target_role, workers):
        return None

    scheduler._cleanup_forked_workers_async = cleanup
    scheduler.delete_workers("teacher")

    assert "teacher" not in scheduler._fork_reservations
    assert "teacher" not in scheduler._workers
    assert "teacher" not in scheduler._colocated_roles


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheduler_type", [LocalScheduler, RayScheduler, SlurmScheduler]
)
@pytest.mark.parametrize(("kill_status", "raises"), [(200, False), (500, True)])
async def test_fork_readiness_failure_retries_only_after_cleanup(
    monkeypatch,
    scheduler_type,
    kill_status,
    raises,
):
    """A failed fork is reaped before either scheduler reserves a new port."""

    class FakeResponse:
        def __init__(self, payload, status=200):
            self.payload = payload
            self.status = status

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self, content_type=None):
            return self.payload

        async def text(self):
            return str(self.payload)

    class FakeSession:
        def __init__(self, kill_status):
            self.allocated_ports = iter((20001, 20002))
            self.calls = []
            self.kill_status = kill_status

        def post(self, url, json):
            self.calls.append((url, json))
            if url.endswith("/reserve_worker_ports"):
                return FakeResponse(
                    {
                        "host": "127.0.0.1",
                        "ports": [next(self.allocated_ports)],
                    }
                )
            if url.endswith("/fork"):
                return FakeResponse({"status": "success", "pid": 1234})
            if url.endswith("/kill_forked_worker"):
                return FakeResponse(
                    {"status": "success"},
                    status=self.kill_status,
                )
            raise AssertionError(f"Unexpected URL: {url}")

    scheduler = object.__new__(scheduler_type)
    scheduler.experiment_name = "test"
    scheduler.trial_name = "retry"
    scheduler.fileroot = None
    scheduler.name_resolve_config = NameResolveConfig(type=None)
    if scheduler_type is LocalScheduler:
        scheduler.log_dir = Path("/tmp")
        owner = WorkerInfo(
            worker=Worker(
                id="actor/0",
                ip="127.0.0.1",
                worker_ports=["19000"],
                engine_ports=[],
            ),
            process=None,
            role="actor",
            gpu_devices=[0],
            created_at=0.0,
            log_file="actor.log",
        )
    elif scheduler_type is RayScheduler:
        owner = RayWorkerInfo(
            worker=Worker(
                id="actor/0",
                ip="127.0.0.1",
                worker_ports=["19000"],
                engine_ports=[],
            ),
            role="actor",
            task_index=0,
        )
    else:
        owner = SlurmWorkerInfo(
            worker=Worker(
                id="actor/0",
                ip="127.0.0.1",
                worker_ports=["19000"],
                engine_ports=[],
            ),
            role="actor",
            slurm_job_id=1,
            task_index=0,
        )
    scheduler._workers = {"actor": [owner]}
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    readiness_results = iter((False, True))

    async def fake_wait_for_ready(*args, **kwargs):
        return next(readiness_results)

    monkeypatch.setattr(scheduler, "_wait_for_fork_ready", fake_wait_for_ready)
    session = FakeSession(kill_status)

    should_raise = raises or scheduler_type is RayScheduler
    if should_raise:
        expected_error = WorkerCleanupError if raises else WorkerCreationError
        error_match = "clean up" if raises else "failed to become ready"
        with pytest.raises(expected_error, match=error_match):
            await scheduler._fork_single_worker(
                session,
                "mopd-teacher",
                0,
                owner,
                "actor",
            )
        if raises:
            assert scheduler._workers["mopd-teacher"][0].worker.worker_ports == [
                "20001"
            ]
            assert scheduler._fork_parent_roles["mopd-teacher"] == "actor"
    else:
        worker_info = await scheduler._fork_single_worker(
            session,
            "mopd-teacher",
            0,
            owner,
            "actor",
        )
        assert worker_info.worker.worker_ports == ["20002"]

    expected_attempts = 1 if should_raise else 2
    assert (
        sum(url.endswith("/reserve_worker_ports") for url, _ in session.calls)
        == expected_attempts
    )
    assert sum(url.endswith("/fork") for url, _ in session.calls) == expected_attempts
    assert sum(url.endswith("/kill_forked_worker") for url, _ in session.calls) == 1
    kill_payload = next(
        payload for url, payload in session.calls if url.endswith("/kill_forked_worker")
    )
    assert kill_payload["release_ports"] is True


@pytest.mark.asyncio
async def test_ray_fork_readiness_rejects_wrong_process_identity(monkeypatch):
    """An unrelated HTTP server on the reserved port is never accepted."""

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self, content_type=None):
            return {"role": "actor", "worker_index": 0}

    class FakeSession:
        def get(self, url, timeout):
            return FakeResponse()

    times = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr("areal.infra.scheduler.ray.time.time", lambda: next(times))

    async def no_sleep(_):
        return None

    monkeypatch.setattr("areal.infra.scheduler.ray.asyncio.sleep", no_sleep)

    ready = await RayScheduler._wait_for_fork_ready(
        FakeSession(),
        "127.0.0.1",
        20001,
        expected_role="mopd-teacher",
        expected_worker_index=0,
        timeout=0.5,
    )

    assert ready is False


def test_ray_graceful_stop_failure_keeps_launcher_and_skips_force_cleanup(monkeypatch):
    """A child-stop failure cannot be hidden by killing its Ray actor owner."""
    scheduler = _scheduler(RayScheduler)
    scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    launcher = object()
    placement_group = object()
    scheduler._launchers = {"actor": [launcher]}
    scheduler._placement_groups = {"actor": placement_group}
    scheduler._destroy_engines_on_workers = lambda workers: None
    scheduler._stop_launchers = lambda role, timeout: ["child kill failed"]
    cleanup_calls = []
    monkeypatch.setattr(
        "areal.infra.scheduler.ray.ray.kill",
        lambda *args, **kwargs: cleanup_calls.append("ray.kill"),
    )
    monkeypatch.setattr(
        "areal.infra.scheduler.ray.remove_placement_group",
        lambda *args, **kwargs: cleanup_calls.append("remove_placement_group"),
    )

    with pytest.raises(WorkerCleanupError, match="child kill failed"):
        scheduler.delete_workers("actor")

    assert scheduler._workers.get("actor")
    assert scheduler._launchers["actor"] == [launcher]
    assert scheduler._placement_groups["actor"] is placement_group
    assert cleanup_calls == []


@pytest.mark.parametrize("failure_stage", ["ray_kill", "placement_group"])
def test_ray_delete_failure_preserves_all_metadata_for_retry(
    monkeypatch, failure_stage
):
    """Ray teardown failures leave role ownership intact for a later retry."""
    scheduler = _scheduler(RayScheduler)
    scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    launcher = object()
    placement_group = object()
    scheduler._launchers = {"actor": [launcher]}
    scheduler._placement_groups = {"actor": placement_group}
    scheduler._destroy_engines_on_workers = lambda workers: None
    scheduler._stop_launchers = lambda role, timeout: []

    if failure_stage == "ray_kill":
        monkeypatch.setattr(
            "areal.infra.scheduler.ray.ray.kill",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kill failed")),
        )
        monkeypatch.setattr(
            "areal.infra.scheduler.ray.remove_placement_group", lambda pg: None
        )
    else:
        monkeypatch.setattr(
            "areal.infra.scheduler.ray.ray.kill", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(
            "areal.infra.scheduler.ray.remove_placement_group",
            lambda pg: (_ for _ in ()).throw(RuntimeError("pg removal failed")),
        )

    with pytest.raises(WorkerCleanupError, match="failed"):
        scheduler.delete_workers("actor")

    assert "actor" in scheduler._workers
    expected_launchers = [launcher] if failure_stage == "ray_kill" else []
    assert scheduler._launchers.get("actor", []) == expected_launchers
    assert scheduler._placement_groups["actor"] is placement_group

    monkeypatch.setattr(
        "areal.infra.scheduler.ray.ray.kill", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "areal.infra.scheduler.ray.remove_placement_group", lambda pg: None
    )
    scheduler.delete_workers("actor")
    assert scheduler._workers == {}
    assert scheduler._launchers == {}
    assert scheduler._placement_groups == {}


def test_slurm_delete_query_failure_without_terminal_state_preserves_metadata(
    monkeypatch,
):
    """Slurm control-plane errors cannot be interpreted as job termination."""
    scheduler = _scheduler(SlurmScheduler)
    scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    scheduler._jobs = {"actor": 42}
    scheduler._job_status_cache = {42: (JobState.RUNNING, 0.0)}
    scheduler._destroy_engines_on_workers = lambda workers: None
    monkeypatch.setattr("areal.infra.scheduler.slurm.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.cancel_jobs",
        lambda **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "scancel")
        ),
    )
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.query_jobs",
        lambda **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "squeue")
        ),
    )
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.query_terminal_state_sacct", lambda _: None
    )

    with pytest.raises(WorkerCleanupError, match="cannot confirm"):
        scheduler.delete_workers("actor")

    assert "actor" in scheduler._workers
    assert scheduler._jobs["actor"] == 42
    assert 42 in scheduler._job_status_cache


def test_slurm_delete_accepts_explicit_terminal_sacct_state(monkeypatch):
    """A failed scancel is harmless only when sacct proves the job is gone."""
    scheduler = _scheduler(SlurmScheduler)
    scheduler._workers.pop("teacher")
    scheduler._colocated_roles = {}
    scheduler._fork_parent_roles = {}
    scheduler._jobs = {"actor": 42}
    scheduler._job_status_cache = {42: (JobState.RUNNING, 0.0)}
    scheduler._destroy_engines_on_workers = lambda workers: None
    monkeypatch.setattr("areal.infra.scheduler.slurm.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.cancel_jobs",
        lambda **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "scancel")
        ),
    )
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.query_jobs",
        lambda **kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "squeue")
        ),
    )
    monkeypatch.setattr(
        "areal.infra.scheduler.slurm.query_terminal_state_sacct",
        lambda _: JobState.CANCELLED,
    )

    scheduler.delete_workers("actor")

    assert scheduler._workers == {}
    assert scheduler._jobs == {}
    assert scheduler._job_status_cache == {}
