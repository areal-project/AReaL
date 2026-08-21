# SPDX-License-Identifier: Apache-2.0

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
