from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
    TeardownCallResult,
    TeardownReport,
)

MODULE = "areal.v2.training_service.controller.controller"


def _make_response(method: str, url: str, *, json=None) -> httpx.Response:
    return httpx.Response(
        200,
        json=json,
        request=httpx.Request(method, url),
    )


def _make_controller(scheduler: MagicMock | None = None) -> GatewayTrainController:
    return GatewayTrainController(
        train_engine="areal.engine.FSDPEngine",
        scheduler=scheduler or MagicMock(),
        config=TrainEngineConfig(
            experiment_name="test-exp",
            trial_name="trial-0",
            backend="fsdp:d2",
            scheduling_spec=(
                SchedulingSpec(
                    cpu=1,
                    gpu=1,
                    mem=1024,
                    port_count=1,
                    cmd="python -m areal.infra.rpc.rpc_server",
                ),
            ),
            admin_api_key="test-admin-key",
            request_timeout=5.0,
            setup_timeout=5.0,
        ),
    )


class _FakeAsyncClient:
    def __init__(self, responses_or_errors):
        self._responses_or_errors = list(responses_or_errors)
        self.get = AsyncMock(side_effect=self._get)
        self.post = AsyncMock(side_effect=self._post)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def _get(self, _url: str):
        next_item = self._responses_or_errors.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    async def _post(self, _url: str, json=None, **kwargs):
        _ = json
        next_item = self._responses_or_errors.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _FakeAiohttpRequest:
    def __init__(self, url: str, events: list[str], failures: set[str]) -> None:
        self._url = url
        self._events = events
        self._failures = failures

    async def __aenter__(self):
        self._events.append(f"start:{self._url}")
        await asyncio.sleep(0)
        if self._url in self._failures:
            self._events.append(f"fail:{self._url}")
            raise RuntimeError(f"failed: {self._url}")
        return self

    async def __aexit__(self, *_args):
        self._events.append(f"done:{self._url}")

    def raise_for_status(self) -> None:
        return None


class _FakeAiohttpSession:
    def __init__(self, events: list[str], failures: set[str] | None = None) -> None:
        self._events = events
        self._failures = failures or set()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def post(self, url: str, **_kwargs) -> _FakeAiohttpRequest:
        return _FakeAiohttpRequest(url, self._events, self._failures)


class TestGatewayTrainControllerInitialization:
    @pytest.mark.asyncio
    async def test_async_initialize_offloads_scheduler_and_uses_async_helpers(self):
        worker0 = MagicMock(ip="127.0.0.1", worker_ports=[18000], id="guard-0")
        worker1 = MagicMock(ip="127.0.0.1", worker_ports=[18001], id="guard-1")

        scheduler = MagicMock()
        scheduler.create_workers.return_value = ["guard-0", "guard-1"]
        scheduler.get_workers.return_value = [worker0, worker1]

        controller = _make_controller(scheduler)
        controller._role = "train-role"

        port_client = _FakeAsyncClient(
            [
                _make_response(
                    "POST",
                    "http://127.0.0.1:18000/alloc_ports",
                    json={"ports": [29500]},
                )
            ]
        )

        async def _run_in_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch("httpx.AsyncClient", return_value=port_client),
            patch(
                f"{MODULE}.asyncio.to_thread", side_effect=_run_in_thread
            ) as mock_to_thread,
            patch.object(
                controller, "_async_set_guards_env", new_callable=AsyncMock
            ) as mock_set_env,
            patch.object(
                controller,
                "_async_fork_on_guard",
                new_callable=AsyncMock,
                side_effect=[
                    ("127.0.0.1", 19001),
                    ("127.0.0.1", 19002),
                    ("127.0.0.1", 18081),
                    ("127.0.0.1", 18082),
                    ("127.0.0.1", 18080),
                ],
            ) as mock_async_fork,
            patch.object(controller, "_fork_on_guard", autospec=True) as mock_sync_fork,
            patch.object(
                controller, "_create_engine_on_worker", new_callable=AsyncMock
            ) as mock_create_engine,
            patch.object(
                controller,
                "_call_worker_engine_endpoint",
                new_callable=AsyncMock,
            ) as mock_call_engine,
            patch.object(
                controller, "_register_in_router", new_callable=AsyncMock
            ) as mock_register,
        ):
            await controller._async_initialize(role="train-role")

        assert mock_to_thread.await_count == 2
        create_call = mock_to_thread.await_args_list[0]
        get_call = mock_to_thread.await_args_list[1]
        assert create_call.args[0] is scheduler.create_workers
        assert get_call.args[0] is scheduler.get_workers
        assert get_call.kwargs == {
            "role": "train-role-guard",
            "timeout": 5,
        }

        mock_set_env.assert_awaited_once()
        assert mock_async_fork.await_count == 5
        mock_sync_fork.assert_not_called()
        assert mock_create_engine.await_count == 2
        assert mock_call_engine.await_count == 4
        mock_register.assert_awaited_once_with(
            "http://127.0.0.1:18081",
            "http://127.0.0.1:18082",
            controller.api_key,
        )

        assert controller._worker_addrs == [
            "http://127.0.0.1:19001",
            "http://127.0.0.1:19002",
        ]
        assert controller._router_addr == "http://127.0.0.1:18081"
        assert controller._model_addr == "http://127.0.0.1:18082"
        assert controller._gateway_addr == "http://127.0.0.1:18080"
        assert controller.api_key is not None
        assert controller.api_key.startswith("ak-train-role-")


class TestGatewayTrainControllerTeardown:
    def test_worker_shutdown_is_global_two_phase(self):
        controller = _make_controller()
        controller._worker_addrs = ["http://worker-0", "http://worker-1"]
        events: list[str] = []
        session = _FakeAiohttpSession(events)

        timeout = MagicMock()
        with (
            patch(
                f"{MODULE}.aiohttp.ClientTimeout", return_value=timeout
            ) as timeout_cls,
            patch(
                f"{MODULE}.aiohttp.ClientSession", return_value=session
            ) as session_cls,
        ):
            report = controller._graceful_shutdown_workers()

        assert report.successful
        timeout_cls.assert_called_once_with(total=controller.config.request_timeout)
        session_cls.assert_called_once_with(timeout=timeout)
        last_awex = max(
            index
            for index, event in enumerate(events)
            if event.startswith("done:") and event.endswith("/awex/teardown")
        )
        first_engine = min(
            index
            for index, event in enumerate(events)
            if event.startswith("start:") and event.endswith("/destroy_engine")
        )
        assert last_awex < first_engine

    def test_forked_service_cleanup_orders_ingress_and_parallelizes_workers(self):
        controller = _make_controller()
        controller._forked_services = [
            ("http://guard-0", "train-worker", 0),
            ("http://guard-1", "train-worker", 1),
            ("http://guard-0", "router", 0),
            ("http://guard-0", "data-proxy", 0),
            ("http://guard-0", "gateway", 0),
        ]
        calls: list[tuple[str, int]] = []
        calls_lock = threading.Lock()
        workers_started = threading.Barrier(2, timeout=1)

        def _kill(_guard_addr: str, role: str, worker_index: int):
            with calls_lock:
                calls.append((role, worker_index))
            if role == "train-worker":
                workers_started.wait()
            return TeardownCallResult("forked-service", role, True)

        with patch.object(controller, "_kill_forked_service", side_effect=_kill):
            results = controller._cleanup_forked_services()

        assert calls[:3] == [("gateway", 0), ("data-proxy", 0), ("router", 0)]
        assert set(calls[3:]) == {("train-worker", 0), ("train-worker", 1)}
        assert len(results) == 5
        assert all(result.success for result in results)

    def test_awex_failure_still_destroys_all_engines_without_false_success_log(self):
        controller = _make_controller()
        controller._worker_addrs = ["http://worker-0", "http://worker-1"]
        failed_url = "http://worker-0/awex/teardown"
        events: list[str] = []
        session = _FakeAiohttpSession(events, failures={failed_url})

        with (
            patch(f"{MODULE}.aiohttp.ClientSession", return_value=session),
            patch(f"{MODULE}.logger") as logger,
        ):
            report = controller._graceful_shutdown_workers()

        assert [(result.phase, result.target) for result in report.failures] == [
            ("awex", "http://worker-0")
        ]
        assert sum(event.endswith("/destroy_engine") for event in events) == 4
        assert not any(
            "destroyed gracefully" in str(call) for call in logger.info.call_args_list
        )
        logger.error.assert_called_once()

    def test_worker_shutdown_orchestration_failure_returns_report(self):
        controller = _make_controller()
        controller._worker_addrs = ["http://worker-0", "http://worker-1"]

        with patch(
            f"{MODULE}.aiohttp.ClientSession",
            side_effect=RuntimeError("session failed"),
        ):
            report = controller._graceful_shutdown_workers()

        assert len(report.awex_results) == 2
        assert len(report.engine_results) == 2
        assert not report.successful
        assert all(
            "session failed" in (result.error or "") for result in report.results
        )

    def test_destroy_reports_failures_after_scheduler_fallback(self):
        scheduler = MagicMock()
        controller = _make_controller(scheduler)
        controller._worker_addrs = ["http://worker-0"]
        controller._guard_addrs = ["http://guard-0"]
        controller._forked_services = [("http://guard-0", "train-worker", 0)]
        controller._service_roles = ["actor-guard"]
        worker_report = TeardownReport(
            engine_results=(
                TeardownCallResult("engine", "http://worker-0", False, "disconnected"),
            )
        )
        fork_result = TeardownCallResult(
            "forked-service", "train-worker/0", False, "kill failed"
        )

        with (
            patch.object(
                controller,
                "_graceful_shutdown_workers",
                return_value=worker_report,
            ),
            patch.object(
                controller,
                "_cleanup_forked_services",
                return_value=(fork_result,),
            ),
            patch.object(
                controller,
                "_verify_guard_cleanup",
                return_value=TeardownCallResult(
                    "guard-drain", "http://guard-0", False, "one child"
                ),
            ),
        ):
            controller.destroy()

        report = controller._last_teardown_report
        assert report is not None
        assert not report.successful
        assert {result.phase for result in report.failures} == {
            "engine",
            "forked-service",
            "guard-drain",
        }
        scheduler.delete_workers.assert_called_once_with(
            role="actor-guard", reverse_order=True
        )

    def test_destroy_preserves_failure_report_when_called_again(self):
        controller = _make_controller()
        failure = TeardownReport(
            engine_results=(
                TeardownCallResult("engine", "http://worker-0", False, "failed"),
            )
        )
        controller._last_teardown_report = failure

        with patch.object(controller, "_cleanup_runtime_state") as cleanup:
            controller.destroy()

        cleanup.assert_not_called()
        assert controller._last_teardown_report is failure

    def test_destroy_reports_process_group_failure_and_keeps_ownership(self):
        controller = _make_controller()
        controller._own_process_group = True

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.destroy_process_group",
                side_effect=RuntimeError("group teardown failed"),
            ),
        ):
            controller.destroy()

        report = controller._last_teardown_report
        assert report is not None
        assert [(result.phase, result.error) for result in report.failures] == [
            ("controller-process-group", "group teardown failed")
        ]
        assert controller._own_process_group is True
