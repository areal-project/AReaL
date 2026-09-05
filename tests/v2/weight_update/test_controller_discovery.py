# SPDX-License-Identifier: Apache-2.0
"""The controller publishes its gateway address so the operator CLI can find it.

`WeightUpdateControllerConfig.port` defaults to 0, so the port is picked inside
`initialize()` and is otherwise known only to that process.
"""

from __future__ import annotations

import pytest

from areal.v2.cli.weight_update.state import ServiceState
from areal.v2.weight_update.controller.config import WeightUpdateControllerConfig
from areal.v2.weight_update.controller.controller import WeightUpdateController


@pytest.fixture(autouse=True)
def isolated_state_dir(tmp_path, monkeypatch):
    # The store re-resolves paths on every call, so the env var is enough.
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    yield


class _FakeProc:
    def __init__(self, pid: int = 4242):
        self.pid = pid

    def poll(self):
        return None


def _controller(**kw) -> WeightUpdateController:
    """A controller with `initialize()`'s subprocess and health wait stubbed.

    Everything else in `initialize()` runs, so the discovery write is exercised
    where it actually sits rather than by calling the helper directly.
    """
    ctl = WeightUpdateController(WeightUpdateControllerConfig(**kw))
    return ctl


@pytest.fixture()
def initialized(monkeypatch):
    def _make(**kw):
        ctl = _controller(**kw)
        monkeypatch.setattr(
            "areal.v2.weight_update.controller.controller.subprocess.Popen",
            lambda *a, **k: _FakeProc(),
        )
        monkeypatch.setattr(
            WeightUpdateController, "_wait_for_health", lambda self: None
        )
        ctl.initialize()
        return ctl

    return _make


class TestDiscoveryFile:
    def test_initialize_publishes_an_address_the_cli_can_load(self, initialized):
        ctl = initialized(admin_api_key="k")
        state = ServiceState.load("default")
        assert state.gateway.url == ctl.gateway_url
        assert state.gateway.url.startswith("http://127.0.0.1:")
        assert state.gateway.url.rsplit(":", 1)[1] != "0"
        assert state.launch_mode == "controller"
        assert state.admin_api_key == "k"
        assert state.gateway.pid == 4242

    def test_service_name_keeps_two_jobs_on_one_machine_apart(self, initialized):
        a = initialized(service_name="job-a")
        b = initialized(service_name="job-b")
        assert ServiceState.load("job-a").gateway.url == a.gateway_url
        assert ServiceState.load("job-b").gateway.url == b.gateway_url
        assert a.gateway_url != b.gateway_url

    def test_destroy_removes_the_file(self, initialized, monkeypatch):
        monkeypatch.setattr(
            "areal.v2.weight_update.controller.controller.kill_process_tree",
            lambda *a, **k: None,
        )
        initialized(service_name="job-a")
        ServiceState.load("job-a")
        # destroy() is the teardown path a training run takes.
        ctl = WeightUpdateController(WeightUpdateControllerConfig(service_name="job-a"))
        ctl.destroy()
        with pytest.raises(FileNotFoundError):
            ServiceState.load("job-a")


class TestNeverBreaksTraining:
    def test_a_failing_state_write_does_not_fail_initialize(
        self, initialized, monkeypatch
    ):
        def boom(self):
            raise OSError("read-only home")

        monkeypatch.setattr(ServiceState, "save", boom)
        ctl = initialized(service_name="job-a")
        assert ctl.gateway_url.startswith("http://")
        with pytest.raises(FileNotFoundError):
            ServiceState.load("job-a")

    def test_a_failing_state_remove_does_not_fail_destroy(
        self, initialized, monkeypatch
    ):
        monkeypatch.setattr(
            "areal.v2.weight_update.controller.controller.kill_process_tree",
            lambda *a, **k: None,
        )
        ctl = initialized(service_name="job-a")

        def boom(cls, service):
            raise OSError("read-only home")

        monkeypatch.setattr(ServiceState, "remove", classmethod(boom))
        ctl.destroy()
        assert ctl.gateway_url == ""
