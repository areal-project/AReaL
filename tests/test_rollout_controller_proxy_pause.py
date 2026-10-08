# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

from areal.infra.controller.rollout_controller import RolloutController


class _RecordingController:
    def __init__(self, proxy_started: bool):
        self.calls: list[str] = []
        self._proxy_started = proxy_started

        class _Dispatcher:
            def __init__(self, calls: list[str]):
                self.calls = calls

            def pause(self):
                self.calls.append("dispatcher.pause")

            def resume(self):
                self.calls.append("dispatcher.resume")

        self.dispatcher = _Dispatcher(self.calls)

    def _collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"workers.{method}")

    def _proxy_collective_rpc(self, method, *args, **kwargs):
        self.calls.append(f"proxy.{method}")

    pause = RolloutController.pause
    resume = RolloutController.resume


def test_pause_and_resume_include_proxy_workers_in_safe_order():
    controller = _RecordingController(proxy_started=True)

    controller.pause()
    controller.resume()

    assert controller.calls == [
        "dispatcher.pause",
        "proxy.pause",
        "workers.pause",
        "workers.resume",
        "proxy.resume",
        "dispatcher.resume",
    ]


def test_pause_and_resume_skip_proxy_rpc_when_not_started():
    controller = _RecordingController(proxy_started=False)

    controller.pause()
    controller.resume()

    assert controller.calls == [
        "dispatcher.pause",
        "workers.pause",
        "workers.resume",
        "dispatcher.resume",
    ]


def test_destroy_bounds_proxy_engine_rpc_timeout():
    calls = []

    class Scheduler:
        async def async_call_engine(self, **kwargs):
            calls.append(kwargs)

        def delete_workers(self, *, role):
            calls.append({"deleted_role": role})

    controller = SimpleNamespace(
        _stop_proxy_gateway=lambda: None,
        _stop_callback_server=lambda: None,
        _dispatcher=None,
        _proxy_started=True,
        proxy_workers=[SimpleNamespace(id="proxy-rollout/0")],
        proxy_addrs=["http://localhost:1"],
        _proxy_engine_name=lambda rank: f"proxy/{rank}",
        _proxy_role="proxy-rollout",
        scheduler=Scheduler(),
        _collective_rpc=lambda method, **kwargs: None,
        _futures_lock=threading.Lock(),
        _pending_futures={},
    )

    RolloutController.destroy(controller)

    assert calls[0]["http_timeout"] == 60.0
    assert calls[0]["method"] == "destroy"
