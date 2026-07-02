from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.api.io_struct import WeightUpdateMeta
from areal.v2.inference_service.controller.controller import RolloutControllerV2
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
    _disk_gateway_save_root,
)
from areal.v2.weight_update.gateway.config import WeightUpdateResult

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


class TestGatewayTrainControllerWeightUpdate:
    @staticmethod
    def _prepare_controller(result=None, error=None):
        controller = _make_controller()
        controller.rollout = MagicMock()
        controller._weight_update_ctrl = MagicMock()
        controller._weight_update_ctrl.update_weights.side_effect = error
        if error is None:
            controller._weight_update_ctrl.update_weights.return_value = result
        return controller

    def test_failed_result_raises_and_keeps_generation_paused(self):
        result = WeightUpdateResult(
            status="error",
            version=1,
            duration_ms=10,
            error="inference load failed",
        )
        controller = self._prepare_controller(result=result)
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="inference load failed"):
            controller.update_weights(meta)

        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_not_called()

    def test_gateway_exception_keeps_generation_paused(self):
        controller = self._prepare_controller(error=TimeoutError("gateway timeout"))
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(TimeoutError, match="gateway timeout"):
            controller.update_weights(meta)

        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_not_called()

    def test_pause_exception_attempts_resume(self):
        controller = self._prepare_controller()
        controller.rollout.pause_generation.side_effect = RuntimeError("pause failed")
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="pause failed"):
            controller.update_weights(meta)

        controller.rollout.continue_generation.assert_called_once_with()

    def test_resume_exception_does_not_mask_pause_exception(self):
        controller = self._prepare_controller()
        controller.rollout.pause_generation.side_effect = RuntimeError("pause failed")
        controller.rollout.continue_generation.side_effect = RuntimeError(
            "resume failed"
        )
        meta = WeightUpdateMeta(
            type="disk",
            path="/tmp/weights",
            version=1,
            clear_checkpoint_after_load=False,
        )

        with pytest.raises(RuntimeError, match="pause failed"):
            controller.update_weights(meta)

    def test_successful_disk_update_removes_versioned_checkpoint(self, tmp_path):
        checkpoint_path = tmp_path / "weight_update_v1"
        checkpoint_path.mkdir()
        result = WeightUpdateResult(
            status="ok",
            version=1,
            duration_ms=10,
        )
        controller = self._prepare_controller(result=result)
        controller._disk_weight_update_root = str(tmp_path)
        meta = WeightUpdateMeta(
            type="disk",
            path=str(checkpoint_path),
            version=1,
            clear_checkpoint_after_load=True,
        )

        controller.update_weights(meta)

        assert not checkpoint_path.exists()
        controller.rollout.continue_generation.assert_called_once_with()

    def test_disk_cleanup_uses_connected_root_not_update_meta_path(self, tmp_path):
        checkpoint_path = tmp_path / "weight_update_v1"
        checkpoint_path.mkdir()
        unrelated_path = tmp_path / "unrelated"
        unrelated_path.mkdir()
        result = WeightUpdateResult(
            status="ok",
            version=1,
            duration_ms=10,
        )
        controller = self._prepare_controller(result=result)
        controller._disk_weight_update_root = str(tmp_path)
        meta = WeightUpdateMeta(
            type="disk",
            path=str(unrelated_path),
            version=1,
            clear_checkpoint_after_load=True,
        )

        controller.update_weights(meta)

        assert not checkpoint_path.exists()
        assert unrelated_path.exists()

    def test_disk_gateway_save_root_matches_versioned_meta_path(self, tmp_path):
        meta = WeightUpdateMeta.from_disk(
            experiment_name="test-exp",
            trial_name="trial-0",
            file_root=str(tmp_path),
        )

        save_root = _disk_gateway_save_root(meta)

        assert os.path.join(save_root, "weight_update_v3") == meta.with_version(3).path

    def test_connect_engine_uses_canonical_disk_save_root(self, tmp_path):
        controller = _make_controller()
        controller._worker_addrs = ["http://train:8000"]
        rollout = RolloutControllerV2.__new__(RolloutControllerV2)
        rollout.config = SimpleNamespace(api_url=None)
        rollout.rollout_alloc = SimpleNamespace(backend="sglang")
        rollout._init_future = None
        rollout._inf_addrs = ["http://infer:9000"]
        meta = WeightUpdateMeta.from_disk(
            experiment_name="test-exp",
            trial_name="trial-0",
            file_root=str(tmp_path),
        )

        with (
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.initialize"
            ),
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.connect"
            ) as mock_connect,
        ):
            controller.connect_engine(rollout, meta)

        expected_root = os.path.dirname(meta.path)
        assert controller._disk_weight_update_root == expected_root
        assert mock_connect.call_args.kwargs["save_path"] == expected_root

    @pytest.mark.parametrize(
        ("backend", "api_url"),
        [("vllm", None), (None, "https://example.com/v1")],
    )
    def test_connect_engine_rejects_unsupported_disk_rollout(self, backend, api_url):
        controller = _make_controller()
        rollout = RolloutControllerV2.__new__(RolloutControllerV2)
        rollout.config = SimpleNamespace(api_url=api_url)
        rollout.rollout_alloc = (
            SimpleNamespace(backend=backend) if backend is not None else None
        )
        rollout._init_future = None
        rollout._inf_addrs = []
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update")

        with (
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.initialize"
            ),
            patch(
                "areal.v2.weight_update.controller.controller."
                "WeightUpdateController.connect"
            ),
            pytest.raises(ValueError, match="local SGLang"),
        ):
            controller.connect_engine(rollout, meta)
