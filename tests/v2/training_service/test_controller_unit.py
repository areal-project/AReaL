from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from areal.api.cli_args import SchedulingSpec, TrainEngineConfig
from areal.infra.utils import process as process_utils
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)

MODULE = "areal.v2.training_service.controller.controller"


def test_disconnect_engine_retains_training_workers_and_reconnects_new_rollout():
    """Rollout replacement releases only the weight-transfer connection."""
    from areal.api.io_struct import WeightUpdateMeta
    from areal.v2.inference_service.controller.controller import RolloutControllerV2
    from areal.v2.weight_update.gateway.config import WeightUpdateResult

    controller = _make_controller()
    controller._ensure_initialized = MagicMock()
    controller._worker_addrs = ["http://train-rank-0"]
    controller._service_roles = ["actor-guard"]
    controller._version = 3
    weights = controller._weight_update_ctrl = MagicMock()
    controller._weight_update_identity = ("old-rollout",)
    controller.disconnect_engine()
    weights.disconnect.assert_called_once_with()
    weights.destroy.assert_called_once_with()
    assert controller._worker_addrs == ["http://train-rank-0"]
    assert controller._service_roles == ["actor-guard"]
    assert controller._version == 3
    assert controller._weight_update_ctrl is None
    assert controller._weight_update_identity is None
    controller.scheduler.delete_workers.assert_not_called()

    rollout = MagicMock(spec=RolloutControllerV2)
    rollout.inference_worker_urls = ["http://new-infer"]
    rollout.inference_guard_addrs = ["http://guard"]
    replacement = MagicMock()
    replacement.update_weights.return_value = WeightUpdateResult(
        status="ok", version=3, duration_ms=1.0
    )
    with (
        patch(
            "areal.v2.weight_update.controller.controller.WeightUpdateController",
            return_value=replacement,
        ),
        patch("requests.post") as post,
    ):
        post.return_value.json.return_value = {"host": "127.0.0.1", "ports": [30000]}
        controller.connect_engine(rollout, WeightUpdateMeta(type="awex", version=3))
        controller.update_weights(WeightUpdateMeta(type="awex", version=3))
    assert replacement.connect.call_args.kwargs["train_worker_urls"] == [
        "http://train-rank-0"
    ]
    assert replacement.connect.call_args.kwargs["inference_worker_urls"] == [
        "http://new-infer"
    ]
    replacement.update_weights.assert_called_once_with(version=3)
    rollout.set_version.assert_called_once_with(3)
    assert controller._service_roles == ["actor-guard"]
    assert controller._version == 3
    controller.scheduler.delete_workers.assert_not_called()


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


@pytest.mark.parametrize(
    "source_dp,target_dp",
    [(2, 2), (2, 4), (4, 2)],
    ids=["same-dp", "grow-dp", "shrink-dp"],
)
def test_full_checkpoint_builds_mapping_from_source_layout(source_dp, target_dp):
    """Every target rank receives the same Controller-owned restore protocol."""
    from areal.api.alloc_mode import ModelAllocation
    from areal.v2.training_service.dp_restore import DPStateRestorePlan

    controller = _make_controller()
    controller.config.backend = f"fsdp:d{target_dp}"
    controller.train_alloc = ModelAllocation.from_str(controller.config.backend)
    controller._worker_addrs = [f"http://rank{rank}" for rank in range(target_dp)]
    calls = []

    async def endpoint(worker, path, **kwargs):
        parameters = kwargs["kwargs"]
        plan = parameters.get("restore_plan")
        if source_dp == target_dp:
            assert plan is None
        else:
            approved = DPStateRestorePlan(**plan)
            assert approved.source_world_size == source_dp
            assert approved.target_world_size == target_dp
            assert approved.checkpoint_generation == "step-3-saved"
            assert approved.identity_sha256 == "a" * 64
            assert approved.policy_version == 3
        calls.append((path, plan))
        rank = controller._worker_addrs.index(worker)
        return {
            "rank": rank,
            "world_size": target_dp,
            "policy_version": 3,
            "backend": "fsdp",
            "identity_sha256": "a" * 64,
            "state_sha256": str(rank) * 64,
        }

    controller._call_worker_engine_endpoint = AsyncMock(side_effect=endpoint)
    reports = controller.full_checkpoint(
        "load",
        path="/checkpoint/step-3-saved",
        backend="fsdp",
        identity_sha256="a" * 64,
        version=3,
        source_world_size=source_dp,
    )
    assert [path for path, _ in calls] == (
        ["/checkpoint/inspect"] * target_dp + ["/checkpoint/load"] * target_dp
    )
    assert all(plan == calls[0][1] for _, plan in calls)
    assert len(reports) == target_dp
    assert controller.get_version() == 3


@pytest.mark.parametrize("release_devices", [None, (0, 1)], ids=["standard", "release"])
def test_destroy_confirms_requested_gpus_after_worker_cleanup(
    monkeypatch, release_devices
):
    """Only explicit local release waits for GPU processes after normal teardown."""
    controller = _make_controller()
    controller._worker_addrs = ["http://rank0", "http://rank1"]
    controller._service_roles = ["actor-guard"]
    controller._graceful_shutdown_workers = MagicMock()
    states = iter([{"GPU-0": [123], "GPU-other": [456]}, {"GPU-other": [456]}])

    def processes():
        assert controller._worker_addrs == []
        assert controller._service_roles == []
        controller._graceful_shutdown_workers.assert_called_once_with()
        controller.scheduler.delete_workers.assert_called_once_with(role="actor-guard")
        return next(states)

    probe = MagicMock(side_effect=processes)
    uuids = MagicMock(return_value=["GPU-0", "GPU-1"])
    sleep = MagicMock()
    monkeypatch.setattr(process_utils, "gpu_processes", probe)
    monkeypatch.setattr(process_utils, "resolve_gpu_uuids", uuids)
    monkeypatch.setattr(process_utils.time, "sleep", sleep)
    controller.destroy(release_devices=release_devices)
    assert controller._worker_addrs == []
    if release_devices is None:
        probe.assert_not_called()
        uuids.assert_not_called()
        sleep.assert_not_called()
    else:
        assert probe.call_count == 2
        uuids.assert_called_once_with([0, 1])
        sleep.assert_called_once_with(0.1)


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


@pytest.fixture
def resident_online_pipeline(tmp_path, monkeypatch):
    """Real CPU data plane; substitute only training/inference GPU execution.

    The DataProxy HTTP server, sessions, RTensor storage, callbacks, workflow
    dispatcher and Trainer step run normally. The CPU worker below is a protocol
    substitute, not an implementation of FSDP or cross-DP tensor resharding.
    """
    import json
    import socket
    import threading
    import time
    from copy import deepcopy
    from types import SimpleNamespace

    import torch
    import uvicorn

    from areal.api import FinetuneSpec, WeightUpdateMeta
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import (
        ActorLifecycleConfig,
        InferenceEngineConfig,
        PPOConfig,
    )
    from areal.elastic.infra import reconfiguration
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.infra.rpc.rtensor import RTensor, clear_fetch_buffer
    from areal.infra.rpc.serialization import deserialize_value, serialize_value
    from areal.infra.staleness_manager import StalenessManager
    from areal.infra.workflow_executor import WorkflowExecutor
    from areal.trainer.ppo.actor import PPOActorControllerV2
    from areal.trainer.rl_trainer import PPOTrainer, _EmptyDataLoader
    from areal.v2.inference_service.controller.controller import RolloutControllerV2
    from areal.v2.inference_service.data_proxy.app import create_app
    from areal.v2.inference_service.data_proxy.config import DataProxyConfig
    from areal.v2.weight_update.gateway.config import WeightUpdateResult

    monkeypatch.setenv("AREAL_SPMD_MODE", "0")
    monkeypatch.setattr("areal.utils.network.gethostip", lambda: "127.0.0.1")
    rollout = RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            consumer_batch_size=4,
            max_concurrent_rollouts=8,
            max_head_offpolicyness=2,
            deterministic_sampling=True,
            dump_to_file=False,
            request_timeout=5,
        ),
        scheduler=MagicMock(n_gpus_per_node=8),
    )
    rollout._start_online_callback_server()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    proxy_config = DataProxyConfig(
        host="127.0.0.1",
        port=port,
        serving_addr=f"127.0.0.1:{port}",
        backend_addr="",
        callback_server_addr=f"http://{rollout.callback_addr}",
    )
    proxy = create_app(proxy_config)
    server = uvicorn.Server(uvicorn.Config(proxy, log_level="error", lifespan="on"))
    server_thread = threading.Thread(
        target=lambda: server.run(sockets=[sock]), daemon=True
    )
    server_thread.start()
    deadline = time.monotonic() + 5
    while not server.started and server_thread.is_alive():
        assert time.monotonic() < deadline, "DataProxy startup timed out"
        time.sleep(0.01)
    assert server.started
    # There is no inference GPU in this test. Keep the proxy/session/storage
    # implementation intact and replace its calls into the inference backend.
    proxy.state.inf_bridge = MagicMock(
        pause=AsyncMock(), resume=AsyncMock(), aclose=AsyncMock()
    )
    address = f"http://127.0.0.1:{port}"
    rollout._gateway_addr = address
    rollout._data_proxy_addrs = [address]
    rollout._inf_addrs = ["http://inference-gpu"]
    rollout._staleness_manager = StalenessManager(rollout, 8, 4, 2)
    executor = rollout._workflow_executor = WorkflowExecutor(
        rollout.config, rollout, rollout.staleness_manager
    )
    executor.initialize()
    stop_waiting = threading.Event()
    wait_results = executor.dispatcher.wait_results

    def bounded_wait(*args, **kwargs):
        if stop_waiting.is_set():
            raise RuntimeError("Resident pipeline verification stopped")
        return wait_results(*args, **kwargs)

    monkeypatch.setattr(executor.dispatcher, "wait_results", bounded_wait)

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = PPOConfig()
    trainer.config.rollout = rollout.config
    trainer.config.actor._version = "v2"
    trainer.config.actor.backend = "fsdp:d2"
    trainer.config.actor.recompute_logprob = False
    trainer.config.actor.use_decoupled_loss = False
    trainer.config.gconfig.n_samples = 1
    trainer._should_offload_rollout = trainer._should_offload_actor = False
    trainer._reconfiguration = None
    trainer._training_lock = threading.RLock()
    trainer._training_active = trainer._training_failed = False
    trainer.actor_alloc = ModelAllocation.from_str("fsdp:d2", name="actor")
    trainer.config.train_dataset.batch_size = 4
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"CPU protocol model placeholder")
    trainer.config.actor.path = str(model)
    trainer.config.actor_lifecycle = ActorLifecycleConfig(
        mode="on_demand",
        state_root=str(tmp_path / "state"),
        actor_devices=(0, 1, 2, 3),
        rollout_devices=(4,),
        checkpoint_estimated_bytes=1,
        checkpoint_free_space_floor=0,
    )
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=12, train_batch_size=4)
    for module in (reconfiguration, process_utils):
        monkeypatch.setattr(module, "resolve_gpu_uuids", lambda devices: list(devices))
        monkeypatch.setattr(module, "gpu_processes", lambda: {})
    trainer.critic = trainer.ref = trainer.teacher = trainer.eval_rollout = None
    trainer.data_controller = None
    trainer.rollout = rollout
    trainer.train_dataloader = _EmptyDataLoader(batch_size=4, steps_per_epoch=3)
    trainer.weight_update_meta = WeightUpdateMeta(
        type="disk", path=str(tmp_path / "weights")
    )
    trainer.saver = MagicMock()
    # These CPU hooks are outside the data/Actor-rebuild operation under test.
    for name in (
        "_save_training_state",
        "_evaluate",
        "_export_and_commit_stats",
        "_save_perf_tracer",
    ):
        monkeypatch.setattr(trainer, name, MagicMock())

    actors, workers, links, consumed = [], {}, [], []

    def snapshot(worker):
        return deepcopy(
            {
                "weight": worker.weight.detach(),
                "optimizer": worker.optimizer.state_dict(),
                "scheduler": worker.scheduler.state_dict(),
                "rng": worker.rng.get_state(),
                "updates": worker.updates,
            }
        )

    def restore(worker, state):
        with torch.no_grad():
            worker.weight.copy_(state["weight"])
        worker.optimizer.load_state_dict(state["optimizer"])
        worker.scheduler.load_state_dict(state["scheduler"])
        worker.rng.set_state(state["rng"])
        worker.updates = state["updates"]

    def create_actor(dp):
        assert not any(worker.running for worker in workers.values())
        config = deepcopy(trainer.config.actor)
        config.backend = f"fsdp:d{dp}"
        actor = PPOActorControllerV2(
            "areal.engine.FSDPPPOActor", config, scheduler=MagicMock()
        )
        weight = torch.nn.Parameter(torch.tensor([0.25]))
        optimizer = torch.optim.Adam([weight], lr=0.01)
        worker = workers[actor] = SimpleNamespace(
            weight=weight,
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9),
            rng=torch.Generator().manual_seed(123),
            updates=0,
            running=True,
        )

        async def initialize(role, ft_spec=None, **kwargs):
            actor._worker_addrs = [
                f"http://actor-{len(actors)}-rank-{r}" for r in range(dp)
            ]
            actor._service_roles = [f"actor-{len(actors)}-guard"]
            actor._workers_ready.set()

        def gateway(path, payload=None, **kwargs):
            assert worker.running
            args = deserialize_value(payload["args"]) if payload else []
            if path == "/ppo/actor/compute_advantages":
                return {"result": serialize_value(args[0])}
            if path == "/ppo/actor/update":
                batch = RTensor.localize(args[0])
                ids = [int(traj["input_ids"][0, 0]) - 100 for traj in batch]
                versions = [
                    int(traj["versions"][traj["loss_mask"].bool()].min())
                    for traj in batch
                ]
                assert len(ids) == 4
                assert all(0 <= actor.get_version() - v <= 2 for v in versions)
                consumed.extend(zip(ids, versions, [actor.get_version()] * 4))
                worker.optimizer.zero_grad()
                target = torch.tensor(ids, dtype=torch.float32).mean() / 10
                target += torch.rand((), generator=worker.rng)
                (worker.weight - target).square().sum().backward()
                worker.optimizer.step()
                worker.updates += 1
            elif path == "/step_lr_scheduler":
                worker.scheduler.step()
            elif path == "/clear_batches":
                clear_fetch_buffer(args[0])
            else:
                assert path == "/set_version", path
            return {"status": "ok"}

        async def checkpoint(addr, endpoint, **kwargs):
            parameters = kwargs["kwargs"]
            rank = actor._worker_addrs.index(addr)
            path = tmp_path / parameters["path"]
            if endpoint == "/checkpoint/save" and rank == 0:
                path.mkdir(exist_ok=True)
                (path / "dcp").mkdir(exist_ok=True)
                (path / "dcp/.metadata").write_bytes(b"CPU worker protocol checkpoint")
                torch.save(snapshot(worker), path / "cpu-worker.pt")
            elif endpoint == "/checkpoint/load":
                restore(worker, torch.load(path / "cpu-worker.pt", weights_only=True))
                plan = parameters["restore_plan"]
                assert plan["target_world_size"] == dp
                assert (
                    plan["source_world_size"]
                    == json.loads((path / "manifest.json").read_text())["world_size"]
                )
            if endpoint == "/checkpoint/save":
                (path / f"runtime-{rank}.json").write_text(
                    json.dumps(
                        {
                            "rank": rank,
                            "world_size": dp,
                            "policy_version": parameters["version"],
                        }
                    )
                )
            return {
                "rank": rank,
                "world_size": dp,
                "policy_version": parameters["version"],
                "backend": parameters["backend"],
                "identity_sha256": parameters["identity_sha256"],
                "state_sha256": str(rank) * 64,
            }

        monkeypatch.setattr(actor, "_async_initialize", initialize)
        monkeypatch.setattr(actor, "_gateway_post", gateway)
        monkeypatch.setattr(actor, "_call_worker_engine_endpoint", checkpoint)
        monkeypatch.setattr(actor, "get_device_stats", MagicMock())
        monkeypatch.setattr(
            actor,
            "_graceful_shutdown_workers",
            lambda: setattr(worker, "running", False),
        )
        actors.append(actor)
        return actor

    def weight_link(_config):
        link = MagicMock()
        link.update_weights.side_effect = lambda *, version: WeightUpdateResult(
            status="ok", version=version, duration_ms=0
        )
        links.append(link)
        return link

    monkeypatch.setattr(
        "areal.v2.weight_update.controller.controller.WeightUpdateController",
        weight_link,
    )

    client = httpx.Client(base_url=address, timeout=5)
    sessions = {}

    def generate(number):
        # Inject a completed GPU response into the real session cache. Session
        # creation, reward sealing, callback, export and RTensor reads use HTTP.
        credentials = client.post(
            "/rl/start_session",
            json={"task_id": f"sample-{number}"},
            headers={"Authorization": f"Bearer {proxy_config.admin_api_key}"},
        )
        credentials.raise_for_status()
        credentials = credentials.json()["sessions"][0]
        session = proxy.state.session_store.get_session(credentials["session_id"])
        interaction = InteractionWithTokenLogpReward(
            _interaction_id=f"sample-{number}",
            messages=[{"role": "user", "content": f"sample-{number}"}],
            output_message_list=[{"role": "assistant", "content": "2"}],
            _cache={
                "input_ids": torch.tensor([[number + 100, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 1]]),
                "logprobs": torch.tensor([[0.0, -0.5]]),
                "versions": torch.tensor([[-1, rollout.get_version()]]),
            },
        )
        session.active_completions[f"sample-{number}"] = interaction
        sessions[number] = credentials
        return credentials

    def reward(number):
        response = client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": f"Bearer {sessions[number]['session_api_key']}"},
        )
        response.raise_for_status()
        assert response.json()["trajectory_ready"]

    monkeypatch.setattr(
        trainer,
        "_create_train_engine",
        lambda config, allocation: create_actor(allocation.parallel.world_size),
    )
    trainer._initialize_reconfigured_actor(trainer.config.actor, ft_spec)
    trainer.actor.connect_engine(rollout, trainer.weight_update_meta)
    trainer._reconfiguration = reconfiguration.TrainerReconfiguration(trainer, ft_spec)
    try:
        yield SimpleNamespace(
            trainer=trainer,
            rollout=rollout,
            proxy=proxy,
            executor=executor,
            actors=actors,
            workers=workers,
            links=links,
            consumed=consumed,
            create_actor=create_actor,
            generate=generate,
            reward=reward,
            snapshot=snapshot,
            stop_waiting=stop_waiting,
        )
    finally:
        for actor in actors:
            if workers[actor].running:
                actor.destroy()
        rollout.destroy()
        client.close()
        server.should_exit = True
        server_thread.join(timeout=5)
        sock.close()
        assert not server_thread.is_alive()


@pytest.mark.parametrize(
    "actor_dps", [(2, 2, 2), (2, 4, 2)], ids=["fixed", "dp-roundtrip"]
)
def test_resident_online_pipeline_continues_after_actor_rebuild(
    resident_online_pipeline, tmp_path, actor_dps, record_property
):
    """Standard Trainer/WFE batches survive 2→4→2 without an Elastic inbox."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from areal.elastic.layout import ExecutionLayout
    from areal.infra.rpc.serialization import serialize_value
    from areal.utils.atomic import json_digest

    pipeline = resident_online_pipeline
    trainer, rollout, executor = pipeline.trainer, pipeline.rollout, pipeline.executor
    manager, dispatcher = rollout.staleness_manager, executor.dispatcher
    original_threads = (
        dispatcher._commit_thread,
        dispatcher._fetch_thread,
        executor.runner.thread,
    )
    original_store = pipeline.proxy.state.session_store

    def wait_until(predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            assert time.monotonic() < deadline, (
                "Resident pipeline did not make progress"
            )
            threading.Event().wait(0.01)

    def step(index):
        trainer._train_step(
            global_step=index,
            steps_per_epoch=3,
            workflow=None,
            workflow_kwargs=None,
            dynamic_filter_fn=None,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )

    # Allow prepare_batch's normal overlap loop to prefetch the second batch.
    # Six completed trajectories then leave two results buffered and two tasks
    # waiting for a reward while the Trainer finishes its first update.
    thread = ThreadPoolExecutor(max_workers=1)
    try:
        first = thread.submit(step, 0)
        wait_until(lambda: len(rollout._online_waiters) == 8)
        for number in range(8):
            pipeline.generate(number)
            if number < 6:
                pipeline.reward(number)
        first.result(timeout=5)
        wait_until(lambda: len(dispatcher._pending_results) == 2)
        assert manager.get_stats().accepted == 6
        assert manager.get_stats().running == 2
        assert trainer.actor.get_version() == rollout.get_version() == 1
        generator = executor.data_generator

        for index, dp in enumerate(actor_dps[1:], start=1):
            actor = trainer.actor
            pending_ids = set(dispatcher._pending_results)
            waiters = tuple(rollout._online_waiters)
            statistics = manager.get_stats()
            capacity = manager.get_capacity()
            previous_state = pipeline.snapshot(pipeline.workers[actor])
            if dp != actor.train_alloc.parallel.world_size:
                receipt = trainer.reconfigure(
                    target=ExecutionLayout(dp, (0, 1, 2, 3), (4,)),
                    request_id=f"step-{index}",
                    expected_generation=index - 1,
                    expected_weights_version=index,
                )
                assert receipt["status"] == "applied"
                assert receipt["generation"] == index
                assert trainer.actor is not actor
                assert not pipeline.workers[actor].running
                replacement = trainer.actor

                torch.testing.assert_close(
                    pipeline.snapshot(pipeline.workers[replacement]),
                    previous_state,
                    rtol=0,
                    atol=0,
                )
                pipeline.links[-2].disconnect.assert_called_once_with()
                pipeline.links[-2].destroy.assert_called_once_with()
                assert (
                    len(
                        pipeline.links[-1].connect.call_args.kwargs["train_worker_urls"]
                    )
                    == dp
                )
            assert executor.data_generator is generator
            assert set(dispatcher._pending_results) == pending_ids
            assert tuple(rollout._online_waiters) == waiters
            assert manager.get_stats() == statistics
            assert manager.get_capacity() == capacity
            assert rollout.staleness_manager is manager
            assert rollout.workflow_executor is executor
            assert pipeline.proxy.state.session_store is original_store
            assert rollout._online_inbox is None
            assert not (tmp_path / "state/inbox").exists()
            assert not executor.is_paused()
            assert trainer.actor.get_version() == rollout.get_version() == index

            current = thread.submit(step, index)
            for number in range(index * 4 + 2, index * 4 + 4):
                pipeline.reward(number)
            if index == 1:
                wait_until(lambda: len(rollout._online_waiters) >= 2)
                for number in range(8, 12):
                    pipeline.generate(number)
                    if number < 10:
                        pipeline.reward(number)
            current.result(timeout=5)
            if index == 1:
                wait_until(lambda: len(dispatcher._pending_results) == 2)

        assert [item[0] for item in pipeline.consumed] == list(range(12))
        assert all(
            0 <= version - generated <= 2 for _, generated, version in pipeline.consumed
        )
        assert (
            trainer.actor.get_version()
            == rollout.get_version()
            == pipeline.proxy.state.version
            == 3
        )
        assert pipeline.workers[trainer.actor].updates == 3
        assert pipeline.workers[trainer.actor].scheduler.last_epoch == 3
        assert manager.get_stats().accepted == 12
        assert manager.get_stats().rejected == 0
        assert all(t.is_alive() for t in original_threads)
        assert (
            dispatcher._commit_thread,
            dispatcher._fetch_thread,
            executor.runner.thread,
        ) == original_threads
        assert len(pipeline.actors) == (3 if actor_dps[1] == 4 else 1)
        record_property("actor_dps", str(actor_dps))
        record_property("consumed_ids", str([item[0] for item in pipeline.consumed]))
        record_property("updates", 3)
        record_property(
            "cpu_worker_state_sha256",
            json_digest(
                serialize_value(pipeline.snapshot(pipeline.workers[trainer.actor]))
            ),
        )
    finally:
        pipeline.stop_waiting.set()
        # Release any waiting workflows before tearing down the real dispatcher.
        for waiter in tuple(rollout._online_waiters):
            waiter.future.get_loop().call_soon_threadsafe(waiter.future.cancel)
        thread.shutdown(wait=True, cancel_futures=True)


def test_native_reconfigurator_switches_during_standard_batch_wait(
    resident_online_pipeline, monkeypatch, record_property
):
    """The real wait callback drives the shared Reconfigurator and Trainer switch."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from tests.test_elastic_online_simulation import _KnownCosts, config, handoff

    from areal.elastic.contracts import EpisodeView, OnlineLayout, RequestView, Turn
    from areal.elastic.infra.adapter import ReconfiguratorAdapter

    p = resident_online_pipeline
    trainer, rollout = p.trainer, p.rollout
    adapter = ReconfiguratorAdapter(
        implementation="reference",
        models=_KnownCosts(),
        run_config=config(),
        candidates=(OnlineLayout(2, 1, 1, 1), OnlineLayout(1, 1, 1, 1)),
        settings=dict(
            minimum_completed_requests=1,
            decision_delay_seconds=1,
            deduplicate_searches=False,
        ),
        replay=dict(history_seconds=120, initial_regime_seconds=120, max_new_tokens=32),
        handoff=replace(
            handoff(),
            mode="cold",
            phase_seconds=tuple(
                (name, 0.001)
                for name in (
                    "checkpoint",
                    "release_actor",
                    "disconnect_weights",
                    "stop_rollout",
                    "rebuild_rollout",
                    "restore_actor",
                    "connect_weights",
                )
            ),
        ),
    )
    trainer._elastic_reconfigurator = adapter
    trainer._control_phase = "idle"
    trainer._bind_reconfigurator()
    observations = trainer._reconfiguration.monitor.online_observations
    clock = [10.0]
    monkeypatch.setattr(observations, "now", lambda *args: clock[0])

    def measured(owner):
        base = observations.observe()
        if clock[0] < 120:
            return replace(base, episodes=(), requests=())
        episodes = tuple(
            EpisodeView(
                f"observed-{i}",
                "fixture",
                150 + i,
                True,
                190 + i,
                (Turn(4, 8),),
                None,
                None,
            )
            for i in range(4)
        )
        requests = tuple(
            RequestView(
                f"r-{i}",
                e.episode_id,
                0,
                4,
                150 + i,
                prefilled=4,
                generated=8,
                first_token_at=151 + i,
                last_token_at=151.07 + i,
                completed_at=151.07 + i,
                deployment=1,
                replica=0,
            )
            for i, e in enumerate(episodes)
        )
        return replace(base, episodes=episodes, requests=requests, consumed_episodes=4)

    adapter.observation_provider = measured
    phases = []
    apply = adapter.environment.request_handoff

    def apply_at_boundary(target):
        phases.append(trainer._control_phase)
        return apply(target)

    monkeypatch.setattr(adapter.environment, "request_handoff", apply_at_boundary)

    def step(index):
        trainer._train_step(
            global_step=index,
            steps_per_epoch=3,
            workflow=None,
            workflow_kwargs=None,
            dynamic_filter_fn=None,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )

    def wait_for(predicate, *, advance=False, task=None):
        deadline = time.monotonic() + 20
        while not predicate():
            if task is not None and task.done():
                task.result()
            assert time.monotonic() < deadline, (
                "Native control loop did not make progress: "
                + str(
                    [
                        (d.get("application_status"), d.get("reason"))
                        for d in adapter.environment.decisions[-2:]
                    ]
                )
            )
            if advance:
                clock[0] += 0.01
            threading.Event().wait(0.02)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        first = pool.submit(step, 0)
        wait_for(lambda: len(rollout._online_waiters) == 8)
        for i in range(8):
            p.generate(i)
            if i < 4:
                p.reward(i)
        first.result(timeout=10)
        monitor = trainer._reconfiguration.monitor
        default = observations.observe()
        assert len(default.episodes) == default.consumed_episodes == 4
        assert all(e.completed_turns for e in default.episodes)
        assert default.requests == ()
        executor, sessions = p.executor, p.proxy.state.session_store
        second = pool.submit(step, 1)
        wait_for(lambda: trainer._control_phase == "waiting")
        clock[0] = 200
        wait_for(
            lambda: adapter.last_receipt is not None
            and trainer.actor_alloc.parallel.dp_size == 1,
            advance=True,
            task=second,
        )
        assert phases == ["waiting"]
        assert adapter.last_receipt["status"] == "applied"
        assert (
            adapter.last_receipt["request"]["reconfigurator_bundle_id"]
            == "builtin-online"
        )
        assert rollout.workflow_executor is executor
        assert p.proxy.state.session_store is sessions
        assert rollout._online_inbox is None
        for i in range(4, 8):
            p.reward(i)
        second.result(timeout=10)
        assert trainer.actor.get_version() == rollout.get_version() == 2
        assert [r[0] for r in p.consumed] == list(range(8))
        assert len(monitor.history) == 2
        assert adapter.reconfigurator.slo.tpot_seconds == 0.05
        record_property("automatic_switch_phase", phases[0])
        record_property("consumed_ids", str([r[0] for r in p.consumed]))
        record_property("weights_version", 2)
    finally:
        p.stop_waiting.set()
        for waiter in tuple(rollout._online_waiters):
            waiter.future.get_loop().call_soon_threadsafe(waiter.future.cancel)
        pool.shutdown(wait=True, cancel_futures=True)
        adapter.close()
