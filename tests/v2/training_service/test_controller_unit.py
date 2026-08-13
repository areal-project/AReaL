from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import torch

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
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


def _make_rtensor(shard_id: str, node_addr: str) -> RTensor:
    return RTensor(
        shard=TensorShardInfo(shard_id=shard_id, node_addr=node_addr),
        data=torch.empty(1, device="meta"),
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


class TestGatewayTrainControllerClearBatches:
    def test_storage_clear_surfaces_failure_and_continues_worker_cleanup(self):
        controller = _make_controller()
        target = {
            "good": _make_rtensor("s-good", "good-node"),
            "bad": _make_rtensor("s-bad", "bad-node"),
        }
        calls = []

        async def fake_clear_node(addr, shard_ids):
            calls.append((addr, shard_ids))
            if addr == "bad-node":
                raise RuntimeError("delete failed")
            return {
                "status": "ok",
                "cleared_count": 1,
                "num_tensors": 0,
                "total_bytes": 0,
            }

        with (
            patch.object(RTensor, "clear_node", new=fake_clear_node),
            patch.object(controller, "_gateway_post") as mock_gateway_post,
            patch(f"{MODULE}.logger.warning") as mock_warning,
            patch(f"{MODULE}.logger.debug") as mock_debug,
        ):
            controller.clear_batches(target)

        assert calls == [
            ("good-node", ["s-good"]),
            ("bad-node", ["s-bad"]),
        ]
        mock_warning.assert_called_once()
        assert "bad-node" in str(mock_warning.call_args)
        mock_debug.assert_called_once()
        assert "good-node" in str(mock_debug.call_args)
        mock_gateway_post.assert_called_once()
        assert controller._pending_clear_shards == {"bad-node": {"s-bad": 1}}

    def test_failed_storage_clear_is_retried_on_next_call(self):
        controller = _make_controller()
        target = {"batch": _make_rtensor("s0", "node-a")}
        next_target = {"batch": _make_rtensor("s1", "node-a")}
        calls = []

        async def fail_then_succeed(addr, shard_ids):
            calls.append((addr, shard_ids))
            if len(calls) == 1:
                raise RuntimeError("delete failed")
            return {"status": "ok", "cleared_count": 1}

        with (
            patch.object(RTensor, "clear_node", new=fail_then_succeed),
            patch.object(controller, "_gateway_post") as mock_gateway_post,
        ):
            controller.clear_batches(target)
            controller.clear_batches(next_target)

        assert calls == [("node-a", ["s0"]), ("node-a", ["s0", "s1"])]
        assert controller._pending_clear_shards == {}
        assert mock_gateway_post.call_count == 2

    def test_second_storage_clear_failure_cleans_workers_then_raises(self):
        controller = _make_controller()
        target = {"batch": _make_rtensor("s0", "node-a")}
        calls = []

        async def fail_clear(addr, shard_ids):
            calls.append((addr, shard_ids))
            raise RuntimeError("delete failed")

        with (
            patch.object(RTensor, "clear_node", new=fail_clear),
            patch.object(controller, "_gateway_post") as mock_gateway_post,
        ):
            controller.clear_batches(target)
            with pytest.raises(RuntimeError, match="two clear_batches calls"):
                controller.clear_batches({})

        assert calls == [("node-a", ["s0"]), ("node-a", ["s0"])]
        assert controller._pending_clear_shards == {}
        assert mock_gateway_post.call_count == 2

    def test_worker_cleanup_failure_preserves_exhausted_storage_state(self):
        controller = _make_controller()
        target = {"batch": _make_rtensor("s0", "node-a")}
        storage_calls = []

        async def fail_clear(addr, shard_ids):
            storage_calls.append((addr, shard_ids))
            raise RuntimeError("delete failed")

        with (
            patch.object(RTensor, "clear_node", new=fail_clear),
            patch.object(
                controller,
                "_gateway_post",
                side_effect=[None, RuntimeError("worker cleanup failed")],
            ) as mock_gateway_post,
        ):
            controller.clear_batches(target)
            with pytest.raises(RuntimeError, match="worker cleanup failed"):
                controller.clear_batches({})

        assert storage_calls == [("node-a", ["s0"]), ("node-a", ["s0"])]
        assert controller._pending_clear_shards == {"node-a": {"s0": 2}}
        assert mock_gateway_post.call_count == 2

    def test_storage_clear_propagates_cancellation(self):
        controller = _make_controller()
        target = {"batch": _make_rtensor("s0", "node-a")}

        async def cancel_clear(_addr, _shard_ids):
            raise asyncio.CancelledError

        with (
            patch.object(RTensor, "clear_node", new=cancel_clear),
            patch.object(controller, "_gateway_post") as mock_gateway_post,
        ):
            with pytest.raises(asyncio.CancelledError):
                controller.clear_batches(target)

        mock_gateway_post.assert_not_called()
        assert controller._pending_clear_shards == {"node-a": {"s0": 0}}

    def test_storage_clear_cancellation_keeps_batch_state_atomic(self):
        controller = _make_controller()
        controller._pending_clear_shards = {"retry-node": {"s-retry": 1}}
        target = {
            "good": _make_rtensor("s-good", "good-node"),
            "cancel": _make_rtensor("s-cancel", "cancel-node"),
        }
        calls = []

        async def mixed_results(addr, shard_ids):
            calls.append((addr, shard_ids))
            if addr == "retry-node":
                raise RuntimeError("second failure")
            if addr == "cancel-node":
                raise asyncio.CancelledError
            return {"status": "ok", "cleared_count": 1}

        with (
            patch.object(RTensor, "clear_node", new=mixed_results),
            patch.object(controller, "_gateway_post") as mock_gateway_post,
        ):
            with pytest.raises(asyncio.CancelledError):
                controller.clear_batches(target)

        assert {addr: sids for addr, sids in calls} == {
            "retry-node": ["s-retry"],
            "good-node": ["s-good"],
            "cancel-node": ["s-cancel"],
        }
        assert controller._pending_clear_shards == {
            "retry-node": {"s-retry": 1},
            "good-node": {"s-good": 0},
            "cancel-node": {"s-cancel": 0},
        }
        mock_gateway_post.assert_not_called()
class TestGatewayTrainControllerWeightUpdateReconnect:
    def test_awex_reconnect_commits_candidate_before_destroying_old_gateway(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        controller = _make_controller()
        controller._role = "actor"
        controller._worker_addrs = ["http://train-0"]

        rollout = MagicMock(spec=RolloutControllerV2)
        rollout.inference_worker_urls = ["http://inference-0"]
        rollout.inference_guard_addrs = ["http://guard-0"]

        first_weight_ctrl = MagicMock()
        recovered_weight_ctrl = MagicMock()
        port_response = MagicMock()
        port_response.json.return_value = {"host": "inference-host", "ports": [12345]}

        with (
            patch(
                "areal.v2.weight_update.controller.controller.WeightUpdateController",
                side_effect=[first_weight_ctrl, recovered_weight_ctrl],
            ) as weight_ctrl_cls,
            patch("requests.post", return_value=port_response),
        ):
            controller.connect_engine(
                rollout,
                SimpleNamespace(type="awex", colocate=False, version=None),
            )
            controller.connect_engine(
                rollout,
                SimpleNamespace(type="awex", colocate=False, version=1),
            )

        first_weight_ctrl.disconnect.assert_called_once()
        assert first_weight_ctrl.disconnect.call_args.kwargs["timeout"] > 0
        first_weight_ctrl.destroy.assert_called_once_with()
        assert recovered_weight_ctrl.connect.call_args.kwargs["pair_name"] == (
            "actor-rollout-v1"
        )
        assert recovered_weight_ctrl.connect.call_args.kwargs["setup_timeout_s"] > 0
        assert recovered_weight_ctrl.connect.call_args.kwargs["rollback_timeout_s"] > 0
        config = weight_ctrl_cls.call_args.args[0]
        assert config.setup_timeout == controller.config.setup_timeout
        assert config.request_timeout == controller.config.request_timeout
        assert config.init_timeout_s == controller.config.request_timeout
        assert config.update_timeout_s == controller.config.request_timeout

    def test_failed_candidate_connect_preserves_old_controller(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        controller = _make_controller()
        controller._role = "actor"
        controller._worker_addrs = ["http://train-0"]

        old_rollout = MagicMock(spec=RolloutControllerV2)
        old_ctrl = MagicMock()
        controller.rollout = old_rollout
        controller._weight_update_ctrl = old_ctrl

        candidate = MagicMock()
        candidate.connect.side_effect = RuntimeError("candidate init failed")
        new_rollout = MagicMock(spec=RolloutControllerV2)
        new_rollout.inference_worker_urls = ["http://inference-0"]
        new_rollout.inference_guard_addrs = ["http://guard-0"]
        port_response = MagicMock()
        port_response.json.return_value = {"host": "inference-host", "ports": [12345]}

        with (
            patch(
                "areal.v2.weight_update.controller.controller.WeightUpdateController",
                return_value=candidate,
            ),
            patch("requests.post", return_value=port_response),
            pytest.raises(RuntimeError, match="candidate init failed"),
        ):
            controller.connect_engine(
                new_rollout,
                SimpleNamespace(type="awex", colocate=False, version=1),
            )

        candidate.destroy.assert_called_once_with()
        old_ctrl.disconnect.assert_not_called()
        old_ctrl.destroy.assert_not_called()
        assert controller._weight_update_ctrl is old_ctrl
        assert controller.rollout is old_rollout

    def test_same_recovery_version_uses_a_unique_candidate_pair_name(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        controller = _make_controller()
        controller._role = "actor"
        controller._worker_addrs = ["http://train-0"]
        old_ctrl = MagicMock()
        old_ctrl.pair_name = "actor-rollout-v1"
        controller._weight_update_ctrl = old_ctrl

        candidate = MagicMock()
        rollout = MagicMock(spec=RolloutControllerV2)
        rollout.inference_worker_urls = ["http://inference-0"]
        rollout.inference_guard_addrs = ["http://guard-0"]
        port_response = MagicMock()
        port_response.json.return_value = {"host": "inference-host", "ports": [12345]}

        with (
            patch(
                "areal.v2.weight_update.controller.controller.WeightUpdateController",
                return_value=candidate,
            ),
            patch("requests.post", return_value=port_response),
        ):
            controller.connect_engine(
                rollout,
                SimpleNamespace(type="awex", colocate=False, version=1),
            )

        pair_name = candidate.connect.call_args.kwargs["pair_name"]
        assert pair_name.startswith("actor-rollout-v1-")
        assert pair_name != old_ctrl.pair_name

    def test_old_teardown_failure_rolls_back_candidate_and_preserves_old(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        controller = _make_controller()
        controller._role = "actor"
        controller._worker_addrs = ["http://train-0"]
        old_rollout = MagicMock(spec=RolloutControllerV2)
        old_ctrl = MagicMock()
        old_ctrl.disconnect.side_effect = RuntimeError("old teardown failed")
        controller.rollout = old_rollout
        controller._weight_update_ctrl = old_ctrl

        candidate = MagicMock()
        new_rollout = MagicMock(spec=RolloutControllerV2)
        new_rollout.inference_worker_urls = ["http://inference-0"]
        new_rollout.inference_guard_addrs = ["http://guard-0"]
        port_response = MagicMock()
        port_response.json.return_value = {"host": "inference-host", "ports": [12345]}

        with (
            patch(
                "areal.v2.weight_update.controller.controller.WeightUpdateController",
                return_value=candidate,
            ),
            patch("requests.post", return_value=port_response),
            pytest.raises(RuntimeError, match="old teardown failed"),
        ):
            controller.connect_engine(
                new_rollout,
                SimpleNamespace(type="awex", colocate=False, version=1),
            )

        candidate.destroy.assert_called_once_with()
        old_ctrl.destroy.assert_not_called()
        assert controller._weight_update_ctrl is old_ctrl
        assert controller.rollout is old_rollout

    def test_update_failure_always_continues_generation(self):
        controller = _make_controller()
        controller.rollout = MagicMock()
        controller._weight_update_ctrl = MagicMock()
        controller._weight_update_ctrl.update_weights.side_effect = RuntimeError(
            "transfer failed"
        )

        with pytest.raises(RuntimeError, match="transfer failed"):
            controller.update_weights(SimpleNamespace(version=3))

        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_called_once_with()
