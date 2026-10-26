# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import torch

from areal.api.cli_args import SchedulingSpec, SchedulingStrategy, TrainEngineConfig
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.v2.inference_service.controller.controller import RolloutControllerV2
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


def _make_controller(
    scheduler: MagicMock | None = None,
    scheduling_strategy: SchedulingStrategy | None = None,
) -> GatewayTrainController:
    return GatewayTrainController(
        train_engine="areal.engine.FSDPEngine",
        scheduler=scheduler or MagicMock(),
        config=TrainEngineConfig(
            experiment_name="test-exp",
            trial_name="trial-0",
            backend="fsdp:d2",
            scheduling_strategy=scheduling_strategy or SchedulingStrategy(),
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
    def test_initialize_returns_future_without_waiting_for_colocated_target(self):
        controller = _make_controller(
            scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout")
        )
        future = MagicMock()
        executor = MagicMock()
        executor.submit.return_value = future
        controller._workers_ready.wait = MagicMock(
            side_effect=AssertionError("colocated initialize must not block")
        )

        with patch(f"{MODULE}.get_executor", return_value=executor):
            result = controller.initialize(role="actor", wait=True)

        assert result is future
        controller._workers_ready.wait.assert_not_called()

    @pytest.mark.parametrize(
        ("scheduling_strategy", "expected_worker_envs"),
        [
            (SchedulingStrategy(), [None, None]),
            (
                SchedulingStrategy(type="colocation", target="rollout"),
                [None, None],
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_async_initialize_offloads_scheduler_and_uses_async_helpers(
        self, scheduling_strategy, expected_worker_envs
    ):
        worker0 = MagicMock(ip="127.0.0.1", worker_ports=[18000], id="guard-0")
        worker1 = MagicMock(ip="127.0.0.1", worker_ports=[18001], id="guard-1")

        scheduler = MagicMock()
        scheduler.n_gpus_per_node = 2
        scheduler.create_workers.return_value = ["guard-0", "guard-1"]
        scheduler.get_workers.return_value = [worker0, worker1]

        controller = _make_controller(scheduler, scheduling_strategy)
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
        data_proxy_cmd = next(
            item.kwargs["raw_cmd"]
            for item in mock_async_fork.await_args_list
            if item.kwargs["role"] == "data-proxy"
        )
        timeout_index = data_proxy_cmd.index("--request-timeout")
        assert (
            float(data_proxy_cmd[timeout_index + 1])
            == controller.config.request_timeout
        )
        assert [
            call.kwargs.get("env") for call in mock_async_fork.await_args_list[:2]
        ] == expected_worker_envs
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


@pytest.mark.parametrize("colocate", [False, True])
def test_connect_engine_awex_allocates_rendezvous_only_for_separation(colocate):
    actor = object.__new__(GatewayTrainController)
    actor._ensure_initialized = MagicMock()
    actor._worker_addrs = ["http://train:8000"]
    actor._role = "actor"
    actor.config = SimpleNamespace(
        admin_api_key="test", log_level="info", request_timeout=30
    )
    rollout = MagicMock(spec=RolloutControllerV2)
    rollout.inference_worker_urls = ["http://infer:8000"]
    rollout.inference_guard_addrs = ["http://infer:9000"]
    response = MagicMock()
    response.json.return_value = {"host": "infer", "ports": [12345]}

    with (
        patch("requests.post", return_value=response) as post,
        patch(
            "areal.v2.weight_update.controller.controller.WeightUpdateController"
        ) as controller,
    ):
        actor.connect_engine(rollout, SimpleNamespace(type="awex", colocate=colocate))

    kwargs = controller.return_value.connect.call_args.kwargs
    assert kwargs["colocate"] is colocate
    if colocate:
        post.assert_not_called()
        assert kwargs["nccl_master_addr"] == ""
        assert kwargs["nccl_master_port"] == 0
    else:
        post.assert_called_once_with(
            "http://infer:9000/alloc_ports", json={"count": 1}, timeout=30
        )
        response.raise_for_status.assert_called_once()
        assert kwargs["nccl_master_addr"] == "infer"
        assert kwargs["nccl_master_port"] == 12345


def test_destroy_waits_for_late_guards_and_cleans_them_once(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    live = set()
    controller = _make_controller(
        scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout")
    )

    def create(*, job):
        entered.set()
        assert release.wait(5)
        live.add(job.role)

    controller.scheduler.create_workers.side_effect = create
    controller.scheduler.delete_workers.side_effect = lambda role: live.remove(role)
    controller._graceful_shutdown_workers = MagicMock()
    with ThreadPoolExecutor(max_workers=3) as pool:
        monkeypatch.setattr(f"{MODULE}.get_executor", lambda *args: pool)
        initialization = controller.initialize(role="actor")
        try:
            assert entered.wait(5)
            destruction = pool.submit(controller.destroy)
            assert controller._shutdown_requested.wait(5)
            assert not destruction.done()
            assert controller._init_future is initialization
            another_destroy = pool.submit(controller.destroy)
        finally:
            release.set()
        destruction.result(timeout=5)
        another_destroy.result(timeout=5)
        initialization.result(timeout=5)

    assert not live
    assert controller._init_future is None
    assert controller._workers_ready.is_set()
    controller.scheduler.get_workers.assert_not_called()
    controller.scheduler.delete_workers.assert_called_once_with(role="actor-guard")


def test_destroy_cancels_queued_initialization_without_creating_guards(monkeypatch):
    release = threading.Event()
    controller = _make_controller(
        scheduling_strategy=SchedulingStrategy(type="colocation", target="rollout")
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        blocker = pool.submit(release.wait)
        monkeypatch.setattr(f"{MODULE}.get_executor", lambda *args: pool)
        try:
            initialization = controller.initialize(role="actor")
            controller.destroy()
            assert initialization.cancelled()
        finally:
            release.set()
            blocker.result(timeout=5)
    assert controller._workers_ready.is_set()
    controller.scheduler.create_workers.assert_not_called()


@pytest.mark.asyncio
async def test_initialization_failure_waits_for_sibling_fork_before_cleanup(
    monkeypatch,
):
    late_started, failure_seen, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    controller = _make_controller()
    controller.scheduler.get_workers.return_value = [
        MagicMock(ip="127.0.0.1", worker_ports=[18000 + i]) for i in range(2)
    ]
    controller._async_set_guards_env = AsyncMock()
    controller._async_wait_for_service = AsyncMock()
    controller._graceful_shutdown_workers = MagicMock()
    controller._kill_forked_service = MagicMock()

    async def post(url, json, **kwargs):
        if url.endswith("/alloc_ports"):
            return _make_response(
                "POST",
                url,
                json={
                    "host": "127.0.0.1",
                    "ports": [19000 + json.get("worker_index", 0)],
                },
            )
        if url.endswith("/fork"):
            if json["worker_index"] == 0:
                await late_started.wait()
                failure_seen.set()
                raise RuntimeError("first fork failed")
            late_started.set()
            await release.wait()
        return _make_response("POST", url, json={"status": "ok"})

    client = SimpleNamespace(post=post)
    controller._get_async_client = AsyncMock(return_value=client)
    task = asyncio.create_task(controller._async_initialize("actor"))
    try:
        await asyncio.wait_for(failure_seen.wait(), 5)
        await asyncio.sleep(0)
        assert not task.done()
        controller.scheduler.delete_workers.assert_not_called()
    finally:
        release.set()
    with pytest.raises(ExceptionGroup, match="initialization failed"):
        await asyncio.wait_for(task, 5)
    controller._kill_forked_service.assert_called_once_with(
        "http://127.0.0.1:18001", "train-worker", 1
    )
    controller.scheduler.delete_workers.assert_called_once_with(role="actor-guard")
    assert controller._forked_services == []
    assert controller._service_roles == []
