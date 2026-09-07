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
    @staticmethod
    def _rollout():
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        rollout = MagicMock(spec=RolloutControllerV2)
        rollout.inference_worker_urls = ["http://inference-0"]
        rollout.inference_guard_addrs = ["http://guard-0"]
        return rollout

    def test_failed_candidate_keeps_old_active_and_pending_cleanup(self):
        controller = _make_controller()
        controller._role = "actor"
        controller._worker_addrs = ["http://train-0"]
        old_ctrl = MagicMock()
        old_ctrl.pair_name = "actor-rollout"
        old_rollout = self._rollout()
        controller._weight_update_ctrl = old_ctrl
        controller.rollout = old_rollout

        candidate = MagicMock()
        candidate.pair_name = "actor-rollout-v1"
        candidate.connect.side_effect = RuntimeError("candidate init failed")
        candidate.destroy.side_effect = RuntimeError("rollback incomplete")
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
                self._rollout(),
                SimpleNamespace(type="awex", version=1),
            )

        candidate.destroy.assert_called_once_with(raise_on_error=True)
        assert controller._weight_update_ctrl is old_ctrl
        assert controller.rollout is old_rollout
        assert controller._stale_weight_update_ctrls == [candidate]

    def test_same_recovery_version_uses_unique_candidate_pair_name(self):
        controller = _make_controller()
        controller._role = "actor"
        controller._worker_addrs = ["http://train-0"]
        old_ctrl = MagicMock()
        old_ctrl.pair_name = "actor-rollout-v1"
        old_ctrl.disconnect.side_effect = RuntimeError("keep stale")
        controller._weight_update_ctrl = old_ctrl
        candidate = MagicMock()
        candidate.pair_name = "candidate"
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
                self._rollout(),
                SimpleNamespace(type="awex", version=1),
            )

        pair_name = candidate.connect.call_args.kwargs["pair_name"]
        assert pair_name.startswith("actor-rollout-v1-")
        assert pair_name != old_ctrl.pair_name
        assert controller._weight_update_ctrl is candidate
        assert controller._stale_weight_update_ctrls == [old_ctrl]

    def test_mutating_update_failure_leaves_generation_paused(self):
        controller = _make_controller()
        controller.rollout = MagicMock()
        controller._weight_update_ctrl = MagicMock()
        controller._weight_update_ctrl.update_weights.side_effect = RuntimeError(
            "transfer failed"
        )

        with pytest.raises(RuntimeError, match="transfer failed"):
            controller.update_weights(SimpleNamespace(version=3))

        controller.rollout.pause_generation.assert_called_once_with()
        controller.rollout.continue_generation.assert_not_called()

    def test_pause_failure_is_marked_pre_mutation_and_skips_transfer(self):
        controller = _make_controller()
        controller.rollout = MagicMock()
        controller.rollout.pause_generation.side_effect = RuntimeError("partial pause")
        controller._weight_update_ctrl = MagicMock()

        with pytest.raises(RuntimeError, match="partial pause") as exc_info:
            controller.update_weights(SimpleNamespace(version=3))

        assert exc_info.value.inference_weights_may_be_mutated is False
        controller._weight_update_ctrl.update_weights.assert_not_called()

    def test_pre_mutation_failure_resumes_generation(self):
        controller = _make_controller()
        controller.rollout = MagicMock()
        controller._weight_update_ctrl = MagicMock()
        error = RuntimeError("preflight failed")
        error.inference_weights_may_be_mutated = False
        controller._weight_update_ctrl.update_weights.side_effect = error

        with pytest.raises(RuntimeError, match="preflight failed"):
            controller.update_weights(SimpleNamespace(version=3))

        controller.rollout.continue_generation.assert_called_once_with()

    def test_shutdown_fallback_covers_active_and_stale_pairs(self):
        controller = _make_controller()

        class FakeWeightController:
            def __init__(self, pair_name: str, failures: int):
                self.pair_name = pair_name
                self.train_worker_urls = [f"http://train-{pair_name}"]
                self.inference_worker_urls = [f"http://infer-{pair_name}"]
                self.failures = failures
                self.destroy_calls = 0

            def disconnect(self, timeout: float) -> None:
                assert timeout == 30.0
                if self.failures:
                    self.failures -= 1
                    raise RuntimeError("gateway disconnect failed")
                self.pair_name = None

            def destroy(self, *, raise_on_error: bool) -> None:
                assert raise_on_error
                self.destroy_calls += 1

        active = FakeWeightController("active", failures=1)
        stale_a = FakeWeightController("stale-a", failures=1)
        stale_b = FakeWeightController("stale-b", failures=0)

        with patch.object(
            controller,
            "_direct_teardown_weight_update_pair",
            return_value=True,
        ) as direct_teardown:
            unresolved = controller._teardown_weight_update_controllers(
                [active, stale_a, stale_b]
            )
            repeated = controller._teardown_weight_update_controllers(
                [active, stale_a, stale_b]
            )

        assert unresolved == repeated == []
        assert [item.args[0] for item in direct_teardown.call_args_list] == [
            "active",
            "stale-a",
        ]
        assert all(item.destroy_calls == 2 for item in (active, stale_a, stale_b))

    def test_direct_shutdown_fallback_sends_pair_json_to_all_endpoints(self):
        controller = _make_controller()
        calls = []

        class FakeResponse:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def post(self, url, json):
                calls.append((url, json))
                return FakeResponse()

        with patch(f"{MODULE}.aiohttp.ClientSession", FakeSession):
            assert controller._direct_teardown_weight_update_pair(
                "pair-a",
                ["http://train-0"],
                ["http://infer-0", "http://infer-1"],
            )

        assert calls == [
            ("http://train-0/awex/teardown", {"pair_name": "pair-a"}),
            ("http://infer-0/awex/teardown", {"pair_name": "pair-a"}),
            ("http://infer-1/awex/teardown", {"pair_name": "pair-a"}),
        ]

    def test_destroy_keeps_workers_alive_until_stale_pair_cleanup_succeeds(self):
        controller = _make_controller()
        stale = MagicMock()
        stale.pair_name = "stale"
        stale.train_worker_urls = ["http://train"]
        stale.inference_worker_urls = ["http://infer"]
        attempts = 0

        def disconnect(*, timeout):
            nonlocal attempts
            assert timeout == 30.0
            attempts += 1
            if attempts == 1:
                raise RuntimeError("busy")
            stale.pair_name = None

        stale.disconnect.side_effect = disconnect
        controller._weight_update_ctrl = stale
        controller._cleanup_runtime_state = MagicMock()

        with patch.object(
            controller,
            "_direct_teardown_weight_update_pair",
            return_value=False,
        ):
            controller.destroy()

        controller._cleanup_runtime_state.assert_not_called()
        assert controller._stale_weight_update_ctrls == [stale]

        controller.destroy()

        controller._cleanup_runtime_state.assert_called_once_with()
