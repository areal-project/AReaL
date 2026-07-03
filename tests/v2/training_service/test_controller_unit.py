from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.infra.utils.http import HttpxAsyncClientCleanup
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
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


class TestGatewayTrainControllerInitialization:
    def test_async_client_closes_on_owner_loop_shutdown(self):
        controller = _make_controller()
        client = MagicMock()
        client.is_closed = False
        close_loops: list[asyncio.AbstractEventLoop] = []

        async def close_client() -> None:
            close_loops.append(asyncio.get_running_loop())
            client.is_closed = True

        client.aclose = AsyncMock(side_effect=close_client)
        loop = asyncio.new_event_loop()
        try:
            with patch(f"{MODULE}.create_httpx_client", return_value=client):
                assert loop.run_until_complete(controller._get_async_client()) is client
        finally:
            loop.close()

        client.aclose.assert_awaited_once_with()
        assert close_loops == [loop]
        assert controller._async_client is None
        assert controller._async_client_loop is None
        assert controller._async_client_cleanup is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("close_fails", [False, True])
    async def test_async_startup_rollback_preserves_primary_on_owner_loop(
        self, close_fails: bool
    ):
        worker0 = MagicMock(ip="127.0.0.1", worker_ports=[18000], id="guard-0")
        worker1 = MagicMock(ip="127.0.0.1", worker_ports=[18001], id="guard-1")
        scheduler = MagicMock()
        scheduler.create_workers.return_value = ["guard-0", "guard-1"]
        scheduler.get_workers.return_value = [worker0, worker1]
        controller = _make_controller(scheduler)
        primary = RuntimeError("guard setup failed")
        cleanup_error = RuntimeError("transport close failed")
        close_loops: list[asyncio.AbstractEventLoop] = []
        client = MagicMock()

        async def close_client() -> None:
            close_loops.append(asyncio.get_running_loop())
            if close_fails:
                raise cleanup_error

        client.aclose = AsyncMock(side_effect=close_client)
        client.post = AsyncMock(side_effect=primary)

        async def _run_in_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with (
            patch(f"{MODULE}.asyncio.to_thread", side_effect=_run_in_thread),
            patch(f"{MODULE}.create_httpx_client", return_value=client),
            patch(f"{MODULE}.register_httpx_client_loop_cleanup"),
            pytest.raises(RuntimeError) as exc_info,
        ):
            await controller._async_initialize(role="train-role")

        assert exc_info.value is primary
        client.aclose.assert_awaited_once_with()
        assert close_loops == [asyncio.get_running_loop()]
        if close_fails:
            assert controller._async_client is client
            assert controller._async_client_cleanup is not None
            assert any(
                "transport close failed" in note
                for note in getattr(primary, "__notes__", [])
            )
        else:
            assert controller._async_client is None
            assert controller._async_client_loop is None
            assert controller._async_client_cleanup is None

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


class TestGatewayTrainControllerLifecycle:
    def test_destroy_retains_async_client_when_close_fails(self):
        controller = _make_controller()
        client = MagicMock()
        owner_loop = MagicMock()
        primary = RuntimeError("async transport close failed")
        cleanup = HttpxAsyncClientCleanup(client, owner_loop)
        controller._async_client = client
        controller._async_client_loop = owner_loop
        controller._async_client_cleanup = cleanup

        with (
            patch(f"{MODULE}.close_httpx_client_from_sync", side_effect=primary),
            pytest.raises(RuntimeError) as exc_info,
        ):
            controller.destroy()

        assert exc_info.value is primary
        assert controller._async_client is client
        assert controller._async_client_loop is owner_loop
        assert controller._async_client_cleanup is cleanup

        with patch(f"{MODULE}.close_httpx_client_from_sync") as close_client:
            controller.destroy()

        close_client.assert_called_once_with(cleanup)
        assert controller._async_client is None
        assert controller._async_client_loop is None
        assert controller._async_client_cleanup is None
