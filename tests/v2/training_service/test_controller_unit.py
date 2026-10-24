from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

MODULE = "areal.v2.training_service.controller.controller"


def test_tracer_flush_uses_bounded_timeout_without_shortening_training_rpcs():
    """Optional tracing never inherits the hour-long training RPC timeout."""
    controller = _make_controller()
    controller.config.request_timeout = 3600
    controller._gateway_post = MagicMock()
    controller.save_perf_tracer(force=True)
    assert controller._gateway_post.call_args.kwargs["timeout"] == 5.0
    assert controller.config.request_timeout == 3600


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


@pytest.mark.parametrize("result_status,result_version", [("error", 1), ("ok", 2)])
def test_unconfirmed_publication_keeps_generation_paused(result_status, result_version):
    from types import SimpleNamespace

    controller = _make_controller()
    controller.rollout = MagicMock()
    controller._weight_update_ctrl = MagicMock()
    controller._weight_update_ctrl.update_weights.return_value = SimpleNamespace(
        status=result_status,
        version=result_version,
    )

    with pytest.raises(RuntimeError, match="not confirmed"):
        controller.update_weights(SimpleNamespace(version=1))

    controller.rollout.pause_generation.assert_called_once()
    controller.rollout.continue_generation.assert_not_called()


@pytest.mark.parametrize("version", [0, 3])
def test_native_publication_can_hold_generation_for_checkpoint_and_cpu_hooks(version):
    """A fresh recovery pair may republish the base model at version zero."""
    from types import SimpleNamespace

    controller = _make_controller()
    controller.rollout = MagicMock()
    controller._weight_update_ctrl = MagicMock()
    controller._weight_update_ctrl.update_weights.return_value = SimpleNamespace(
        status="ok", version=version, duration_ms=0
    )
    controller.update_weights(SimpleNamespace(version=version), resume_generation=False)
    controller._weight_update_ctrl.update_weights.assert_called_once_with(
        version=version
    )
    controller.rollout.set_version.assert_called_once_with(version)
    controller.rollout.continue_generation.assert_not_called()


def test_awex_disconnect_failure_does_not_destroy_training_peers():
    controller = _make_controller()
    controller._weight_update_ctrl = MagicMock()
    controller._weight_update_ctrl.disconnect.side_effect = RuntimeError(
        "inference cleanup unknown"
    )
    controller._graceful_shutdown_workers = MagicMock()

    with pytest.raises(RuntimeError, match="cleanup unknown"):
        controller._cleanup_runtime_state()

    controller._graceful_shutdown_workers.assert_not_called()
    controller._weight_update_ctrl.destroy.assert_not_called()
    controller.scheduler.delete_workers.assert_not_called()


def test_initialization_timeout_retains_generation_for_later_cleanup():
    from concurrent.futures import Future

    controller = _make_controller()
    future = Future()
    future.set_running_or_notify_cancel()
    controller._init_future = future
    controller._cleanup_runtime_state = MagicMock()
    with pytest.raises(RuntimeError, match="still running"):
        controller.quiesce_initialization(timeout=0.001)
    assert controller._init_future is future
    assert controller._shutdown_requested.is_set()
    controller._cleanup_runtime_state.assert_not_called()
    future.set_result(None)
    controller.quiesce_initialization(timeout=0.001)
    assert controller._init_future is None


def test_completed_initialization_timeout_does_not_prevent_cleanup():
    from concurrent.futures import Future

    controller = _make_controller()
    future = Future()
    future.set_exception(TimeoutError("Worker startup timed out"))
    controller._init_future = future
    controller._cleanup_runtime_state = MagicMock()
    controller.destroy()
    assert controller._init_future is None
    controller._cleanup_runtime_state.assert_called_once_with(training_stopped=False)


def test_stopped_training_cleanup_requires_surviving_inference_ack():
    controller = _make_controller()
    control = controller._weight_update_ctrl = MagicMock()
    control.disconnect.side_effect = RuntimeError("inference unknown")
    controller._graceful_shutdown_workers = MagicMock()
    with pytest.raises(RuntimeError, match="inference unknown"):
        controller.destroy(training_stopped=True)
    control.disconnect.assert_called_once_with(training_stopped=True)
    assert controller._weight_update_ctrl is control
    controller.scheduler.delete_workers.assert_not_called()
    control.disconnect.side_effect = None
    controller.destroy(training_stopped=True)
    assert controller._weight_update_ctrl is None
    controller._graceful_shutdown_workers.assert_not_called()


def test_failed_cpu_group_destroy_keeps_ownership_until_confirmed(monkeypatch):
    import torch.distributed as dist

    controller = _make_controller()
    controller._own_process_group = True
    controller._graceful_shutdown_workers = MagicMock()
    initialized = MagicMock(return_value=True)
    destroy = MagicMock(side_effect=RuntimeError("group still alive"))
    monkeypatch.setattr(dist, "is_initialized", initialized)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="group still alive"):
        controller.destroy()
    assert controller._own_process_group

    def finish():
        initialized.return_value = False

    destroy.side_effect = finish
    controller.destroy()
    assert not controller._own_process_group


def test_recovery_reconnect_reuses_live_pair_and_can_publish():
    from areal.api.io_struct import WeightUpdateMeta
    from areal.v2.inference_service.controller.controller import RolloutControllerV2
    from areal.v2.weight_update.gateway.config import WeightUpdateResult

    controller = _make_controller()
    controller._ensure_initialized = MagicMock()
    controller._worker_addrs = ["http://train"]
    rollout = MagicMock(spec=RolloutControllerV2)
    rollout.inference_worker_urls = ["http://infer"]
    rollout.inference_guard_addrs = ["http://guard"]
    ctrl = MagicMock()
    ctrl.update_weights.return_value = WeightUpdateResult(
        status="ok", version=5, duration_ms=1.0
    )
    with (
        patch(
            "areal.v2.weight_update.controller.controller.WeightUpdateController",
            return_value=ctrl,
        ) as factory,
        patch("requests.post") as post,
    ):
        post.return_value.json.return_value = {"host": "127.0.0.1", "ports": [30000]}
        controller.connect_engine(rollout, WeightUpdateMeta(type="awex"))
        # RecoverHandler reconnects after loading an existing checkpoint.
        controller.connect_engine(rollout, WeightUpdateMeta(type="awex", version=5))
        controller.update_weights(WeightUpdateMeta(type="awex", version=5))

    factory.assert_called_once()
    ctrl.connect.assert_called_once()
    ctrl.disconnect.assert_not_called()
    ctrl.update_weights.assert_called_once_with(version=5)
    rollout.set_version.assert_called_once_with(5)
    rollout.continue_generation.assert_called_once()


def test_topology_change_keeps_old_handle_when_disconnect_is_uncertain():
    from areal.api.io_struct import WeightUpdateMeta
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    controller = _make_controller()
    controller._ensure_initialized = MagicMock()
    old = MagicMock()
    old.disconnect.side_effect = RuntimeError("uncertain disconnect")
    controller._weight_update_ctrl = old
    rollout = MagicMock(spec=RolloutControllerV2)
    rollout.inference_worker_urls = ["http://new-infer"]
    with pytest.raises(RuntimeError, match="uncertain disconnect"):
        controller.connect_engine(rollout, WeightUpdateMeta(type="awex"))
    assert controller._weight_update_ctrl is old
    old.destroy.assert_not_called()


def test_full_checkpoint_preflight_failure_never_starts_collective_load():
    controller = _make_controller()
    controller._ensure_initialized = MagicMock()
    controller._worker_addrs = ["http://rank0", "http://rank1"]
    events = []

    async def endpoint(worker, path, **kwargs):
        events.append((worker, path))
        if worker.endswith("rank1"):
            raise ValueError("Rank sidecar does not match")
        return {}

    controller._call_worker_engine_endpoint = AsyncMock(side_effect=endpoint)
    with pytest.raises(ValueError, match="sidecar"):
        controller.full_checkpoint(
            "load",
            path="/checkpoint",
            backend="fsdp",
            identity_sha256="a" * 64,
            version=2,
        )
    assert set(events) == {
        (worker, "/checkpoint/inspect") for worker in controller._worker_addrs
    }
    assert controller.get_version() == 0


def test_full_checkpoint_mapped_load_preflights_all_target_ranks():
    """The approved plan reaches every target and target digests agree after load."""
    controller = _make_controller()
    controller._ensure_initialized = MagicMock()
    controller._worker_addrs = ["http://rank0", "http://rank1"]
    events = []
    plan = {"source_world_size": 4, "target_world_size": 2}

    async def endpoint(worker, path, **kwargs):
        assert kwargs["kwargs"]["restore_plan"] == plan
        events.append(path)
        rank = int(worker[-1])
        return {
            "rank": rank,
            "world_size": 2,
            "policy_version": 3,
            "backend": "fsdp",
            "identity_sha256": "a" * 64,
            "state_sha256": str(rank) * 64,
        }

    controller._call_worker_engine_endpoint = AsyncMock(side_effect=endpoint)
    reports = controller.full_checkpoint(
        "load",
        path="/checkpoint",
        backend="fsdp",
        identity_sha256="a" * 64,
        version=3,
        restore_plan=plan,
    )
    assert events == [
        "/checkpoint/inspect",
        "/checkpoint/inspect",
        "/checkpoint/load",
        "/checkpoint/load",
    ]
    assert len(reports) == 2 and controller.get_version() == 3


def test_version_rpc_failure_does_not_advance_controller_version():
    controller = _make_controller()
    controller._gateway_post = MagicMock(
        side_effect=RuntimeError("partial version acknowledgement")
    )
    with pytest.raises(RuntimeError, match="partial version"):
        controller.set_version(2)
    assert controller.get_version() == 0


def test_full_checkpoint_excludes_training_mutations_until_all_ranks_complete():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    controller = _make_controller()
    controller._ensure_initialized = MagicMock()
    controller._worker_addrs = ["http://rank0", "http://rank1"]
    controller._gateway_addr = "http://gateway"
    entered, release, attempted, mutated = (threading.Event() for _ in range(4))
    events = []

    async def endpoint(worker, path, **kwargs):
        if path.endswith("inspect"):
            entered.set()
            assert release.wait(timeout=5)
        else:
            events.append("checkpoint")
        rank = int(worker[-1])
        return {
            "rank": rank,
            "world_size": 2,
            "policy_version": 1,
            "backend": "fsdp",
            "identity_sha256": "a" * 64,
            "state_sha256": str(rank) * 64,
        }

    def post(*args, **kwargs):
        events.append("training")
        mutated.set()
        response = MagicMock(status_code=200)
        response.json.return_value = {}
        return response

    def train():
        attempted.set()
        controller._gateway_post("/optimizer_step")

    controller._call_worker_engine_endpoint = AsyncMock(side_effect=endpoint)
    with (
        patch("requests.post", side_effect=post),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        checkpoint = pool.submit(
            controller.full_checkpoint,
            "save",
            path="/checkpoint",
            backend="fsdp",
            identity_sha256="a" * 64,
            version=1,
        )
        assert entered.wait(timeout=2)
        mutation = pool.submit(train)
        assert attempted.wait(timeout=2)
        try:
            assert not mutated.wait(timeout=0.05)
        finally:
            release.set()
        checkpoint.result(timeout=5)
        mutation.result(timeout=5)
    assert events == ["checkpoint", "checkpoint", "training"]
