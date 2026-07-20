from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("ray")

from areal.infra.rpc import ray_http_worker_manager as rhwm  # noqa: E402


class FakeProcess:
    def __init__(self, *, pid: int = 1234, poll_result: int | None = None):
        self.pid = pid
        self.returncode = poll_result
        self._poll_result = poll_result
        self.killed = False

    def poll(self) -> int | None:
        return self._poll_result

    def kill(self) -> None:
        self.killed = True


def test_launch_uses_worker_indexed_log_file_and_merged_log_label(
    monkeypatch, tmp_path
):
    manager = rhwm._RayHTTPWorkerManagerImpl()
    process = FakeProcess()
    run_with_logs = Mock(return_value=process)

    monkeypatch.setattr(rhwm, "run_with_streaming_logs", run_with_logs)
    monkeypatch.setattr(
        rhwm._RayHTTPWorkerManagerImpl,
        "_wait_ready",
        lambda self, timeout: None,
    )
    monkeypatch.setattr(
        rhwm._RayHTTPWorkerManagerImpl,
        "get_node_id",
        lambda self: "node-1",
    )
    monkeypatch.setattr(rhwm.ray.util, "get_node_ip_address", lambda: "10.0.0.8")

    launch_info = manager.launch(
        module="example.module",
        host="0.0.0.0",
        port=8080,
        experiment_name="exp",
        trial_name="trial",
        role="proxy",
        worker_index=1,
        name_resolve_type="nfs",
        nfs_record_root="/tmp/records",
        etcd3_addr="",
        fileroot=str(tmp_path),
        env={"A": "B"},
        startup_timeout=1.0,
    )

    assert launch_info["host"] == "10.0.0.8"
    assert launch_info["port"] == 8080
    assert launch_info["node_id"] == "node-1"
    assert launch_info["log_file"].endswith("proxy-1.log")

    run_with_logs.assert_called_once()
    _, log_file, merged_log, log_label = run_with_logs.call_args.args
    assert log_file.name == "proxy-1.log"
    assert merged_log.name == "merged.log"
    assert log_label == "proxy-1"
    assert run_with_logs.call_args.kwargs["env"]["A"] == "B"


def test_get_advertised_host_prefers_ray_node_ip_for_wildcard(monkeypatch):
    manager = rhwm._RayHTTPWorkerManagerImpl()
    monkeypatch.setattr(rhwm.ray.util, "get_node_ip_address", lambda: "10.0.0.9")

    assert manager._get_advertised_host("0.0.0.0") == "10.0.0.9"
    assert manager._get_advertised_host("::") == "10.0.0.9"


def test_get_advertised_host_falls_back_to_network_helper(monkeypatch):
    manager = rhwm._RayHTTPWorkerManagerImpl()

    def raise_ray_error() -> str:
        raise RuntimeError("ray node ip unavailable")

    monkeypatch.setattr(rhwm.ray.util, "get_node_ip_address", raise_ray_error)
    monkeypatch.setattr(rhwm, "gethostip", lambda: "10.0.0.10")

    assert manager._get_advertised_host("0.0.0.0") == "10.0.0.10"


def test_get_advertised_host_keeps_explicit_host(monkeypatch):
    manager = rhwm._RayHTTPWorkerManagerImpl()
    get_node_ip = Mock(return_value="10.0.0.11")
    monkeypatch.setattr(rhwm.ray.util, "get_node_ip_address", get_node_ip)

    assert manager._get_advertised_host("192.0.2.10") == "192.0.2.10"
    get_node_ip.assert_not_called()


def test_wait_ready_raises_with_log_tail_when_process_exits(tmp_path):
    log_file = tmp_path / "worker.log"
    log_file.write_text("first line\nlast line\n")

    manager = rhwm._RayHTTPWorkerManagerImpl()
    manager._process = FakeProcess(poll_result=12)
    manager._log_file = str(log_file)

    with pytest.raises(RuntimeError) as exc_info:
        manager._wait_ready(startup_timeout=1.0)

    message = str(exc_info.value)
    assert "HTTP server exited during startup with code 12" in message
    assert "last line" in message


def test_wait_ready_timeout_includes_last_health_error_and_log_tail(
    monkeypatch, tmp_path
):
    log_file = tmp_path / "worker.log"
    log_file.write_text("startup details\n")

    manager = rhwm._RayHTTPWorkerManagerImpl()
    manager._process = FakeProcess()
    manager._host = "127.0.0.1"
    manager._port = 8080
    manager._log_file = str(log_file)

    times = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(rhwm.time, "time", lambda: next(times))
    monkeypatch.setattr(rhwm.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        rhwm.requests,
        "get",
        Mock(side_effect=RuntimeError("connection refused")),
    )

    with pytest.raises(TimeoutError) as exc_info:
        manager._wait_ready(startup_timeout=0.1)

    message = str(exc_info.value)
    assert "connection refused" in message
    assert "startup details" in message


def test_destroy_without_actor_exit_kills_process_tree(monkeypatch):
    manager = rhwm._RayHTTPWorkerManagerImpl()
    manager._process = FakeProcess(pid=4321)
    kill_process_tree = Mock()
    exit_actor = Mock()

    monkeypatch.setattr(rhwm, "kill_process_tree", kill_process_tree)
    monkeypatch.setattr(rhwm.ray.actor, "exit_actor", exit_actor)

    manager.destroy(exit_actor=False)

    kill_process_tree.assert_called_once_with(4321, timeout=5, graceful=True)
    exit_actor.assert_not_called()
    assert manager._process is None
