# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AgentController.

All Guard HTTP interactions are mocked — no real processes or servers.
Tests cover: initialize, destroy, scale_up, scale_down, and error handling.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from areal.api.cli_args import AgentConfig, SchedulingSpec
from areal.experimental.agent_service.controller.controller import AgentController

CTRL = "areal.experimental.agent_service.controller.controller"


@dataclass
class _FakeWorker:
    id: str
    ip: str
    worker_ports: list[str]
    engine_ports: list[str]


def _make_scheduler(*guard_specs: tuple[str, str]) -> MagicMock:
    """Return a mock Scheduler whose get_workers returns _FakeWorkers."""
    workers = [
        _FakeWorker(id=f"agent-guard/{i}", ip=ip, worker_ports=[port], engine_ports=[])
        for i, (ip, port) in enumerate(guard_specs)
    ]
    scheduler = MagicMock()
    scheduler.get_workers.return_value = workers
    return scheduler


def _mock_alloc_ports_response(host: str, ports: list[int]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "ports": ports}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_fork_response(host: str, pid: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success", "host": host, "pid": pid}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_kill_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "success"}
    resp.text = '{"status": "success"}'
    return resp


def _mock_register_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def _mock_health_response(active_sessions: int = 0) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "active_sessions": active_sessions}
    return resp


@pytest.fixture()
def config():
    return AgentConfig(
        agent_cls_path="my.Agent",
        admin_api_key="test-key",
    )


def _setup_mock_requests(mock_requests, port_start=9001):
    port_counter = iter(range(port_start, port_start + 100))

    def mock_post(url, **kwargs):
        if "/alloc_ports" in url:
            return _mock_alloc_ports_response("10.0.0.1", [next(port_counter)])
        if "/fork" in url:
            return _mock_fork_response("10.0.0.1", 100)
        if "/register" in url:
            return _mock_register_response()
        if "/kill_forked_worker" in url:
            return _mock_kill_response()
        if "/unregister" in url:
            return _mock_register_response()
        return MagicMock(status_code=404)

    mock_requests.post = mock_post
    mock_requests.get = lambda url, **kw: _mock_health_response()
    mock_requests.RequestException = Exception


class TestConstruction:
    def test_construction(self, config):
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}


class TestInitialize:
    @patch(f"{CTRL}.requests")
    def test_initialize_forks_router_pairs_gateway(self, mock_requests, config):
        """Initialize should create guards via scheduler, then fork services."""
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"), ("10.0.0.2", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        scheduler.create_workers.assert_called_once()
        scheduler.get_workers.assert_called_once()

        assert "http://" in ctrl.router_addr
        assert "http://" in ctrl.gateway_addr
        assert len(ctrl.pairs) == 1
        assert len(ctrl._forked_services) == 4

        ctrl.destroy()

    @patch(f"{CTRL}.requests")
    def test_initialize_uses_scheduling_spec_env_vars(self, mock_requests):
        fork_payloads = []

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                fork_payloads.append(kwargs["json"])
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        config = AgentConfig(
            agent_cls_path="my.Agent",
            admin_api_key="test-key",
            scheduling_spec=(
                SchedulingSpec(env_vars={"ANTHROPIC_API_KEY": "test-anthropic-key"}),
            ),
        )
        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.tasks[0].env_vars == {"ANTHROPIC_API_KEY": "test-anthropic-key"}
        assert (
            job.tasks[0].cmd
            == f"{sys.executable} -m areal.experimental.agent_service.guard"
        )
        assert all(
            payload.get("env") == {"ANTHROPIC_API_KEY": "test-anthropic-key"}
            for payload in fork_payloads
        )

        ctrl.destroy()


class TestScaleUp:
    @patch(f"{CTRL}.requests")
    def test_scale_up_adds_pairs(self, mock_requests, config):
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl.pairs) == 1

        created = ctrl.scale_up(3)
        assert created == [1, 2, 3]
        assert len(ctrl.pairs) == 4

        ctrl.destroy()

    @patch(f"{CTRL}.requests")
    def test_scale_up_round_robins_guards(self, mock_requests, config):
        guards_called: list[str] = []

        def mock_post(url, **kwargs):
            if "/alloc_ports" in url:
                guards_called.append(url.split("/alloc_ports")[0])
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/register" in url:
                return _mock_register_response()
            if "/kill_forked_worker" in url:
                return _mock_kill_response()
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        scheduler = _make_scheduler(("g0", "8090"), ("g1", "8091"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        guards_called.clear()

        ctrl.scale_up(4)

        g0_calls = [g for g in guards_called if "g0" in g]
        g1_calls = [g for g in guards_called if "g1" in g]
        assert len(g0_calls) == 4
        assert len(g1_calls) == 4

        ctrl.destroy()


class TestScaleDown:
    @patch(f"{CTRL}.requests")
    def test_scale_down_removes_newest_first(self, mock_requests, config):
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        ctrl.scale_up(2)
        assert len(ctrl.pairs) == 3

        removed = ctrl.scale_down(2)
        assert set(removed) == {2, 1}
        assert len(ctrl.pairs) == 1
        assert 0 in ctrl.pairs

        ctrl.destroy()


class TestDestroy:
    @patch(f"{CTRL}.requests")
    def test_destroy_clears_everything(self, mock_requests, config):
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert len(ctrl._forked_services) > 0

        ctrl.destroy()
        assert ctrl.router_addr == ""
        assert ctrl.gateway_addr == ""
        assert ctrl.pairs == {}
        assert ctrl._forked_services == []
        scheduler.delete_workers.assert_called()

    @patch(f"{CTRL}.requests")
    def test_destroy_tolerates_kill_errors(self, mock_requests, config):
        kill_count = 0

        def mock_post(url, **kwargs):
            nonlocal kill_count
            if "/alloc_ports" in url:
                return _mock_alloc_ports_response("10.0.0.1", [9001])
            if "/fork" in url:
                return _mock_fork_response("10.0.0.1", 100)
            if "/kill_forked_worker" in url:
                kill_count += 1
                raise ConnectionError("Guard down")
            return MagicMock(status_code=404)

        mock_requests.post = mock_post
        mock_requests.get = lambda url, **kw: _mock_health_response()
        mock_requests.RequestException = Exception

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        ctrl.destroy()
        assert kill_count == 4
        assert ctrl._forked_services == []


class TestDrain:
    @patch(f"{CTRL}.requests")
    def test_scale_down_waits_for_drain(self, mock_requests, config):
        """scale_down should poll DataProxy health until active_sessions reaches 0."""
        _setup_mock_requests(mock_requests)
        health_call_count = 0

        def mock_get(url, **kwargs):
            nonlocal health_call_count
            health_call_count += 1
            if "/health" in url and health_call_count <= 5:
                return _mock_health_response(active_sessions=2)
            return _mock_health_response(active_sessions=0)

        mock_requests.get = mock_get

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        health_call_count = 0
        with patch(f"{CTRL}.time") as mock_time:
            mock_time.monotonic = time.monotonic
            mock_time.sleep = MagicMock()
            ctrl.scale_down(1)

        assert len(ctrl.pairs) == 0
        assert health_call_count > 1

        ctrl.destroy()

    @patch(f"{CTRL}.requests")
    def test_drain_uses_default_timeout(self, mock_requests, config):
        _setup_mock_requests(mock_requests)
        get_count = 0

        def counting_get(url, **kwargs):
            nonlocal get_count
            get_count += 1
            return _mock_health_response(active_sessions=5)

        mock_requests.get = counting_get

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()

        pre_get_count = get_count
        ctrl.scale_down(1)
        drain_gets = get_count - pre_get_count
        assert drain_gets > 0

        ctrl.destroy()


class TestHealthMonitor:
    @patch(f"{CTRL}.requests")
    def test_health_monitor_starts_and_stops(self, mock_requests, config):
        _setup_mock_requests(mock_requests)

        scheduler = _make_scheduler(("10.0.0.1", "8090"))
        ctrl = AgentController(config=config, scheduler=scheduler)
        ctrl.initialize()
        assert ctrl._health_thread is not None
        assert ctrl._health_thread.is_alive()

        ctrl.destroy()
        assert ctrl._health_thread is None
