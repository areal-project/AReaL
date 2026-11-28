# SPDX-License-Identifier: Apache-2.0

"""Admission and observation tests; no CUDA or inference servers are required."""

import queue
import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.infra.controller.rollout_controller import (
    RolloutController,
    _RemoteRolloutTaskInput,
)
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.workflow_executor import _RolloutResult
from areal.utils.rollout_metrics import ConsumedMetrics


@pytest.fixture
def controller():
    """Construct only the state needed by isolated stream controller tests."""
    value = RolloutController.__new__(RolloutController)
    value.workers = [SimpleNamespace(id=f"rollout-{rank}") for rank in range(4)]
    value._worker_role = "rollout"
    value._current_worker_idx = 0
    value._stream_lock = threading.RLock()
    value._stream_collecting = False
    value._stream_events = queue.Queue()
    value._stream_active = defaultdict(int)
    value._stream_closed = set()
    value._stream_drained = set()
    value._stream_task_workers = {}
    value._stream_failure = None
    value._backend_router = None
    value._proxy_started = False
    value.scheduler = SimpleNamespace(async_call_engine=AsyncMock())
    value.consumed_metrics = ConsumedMetrics()
    return value


def test_stream_drain_active_workflow_keeps_replica_running(controller):
    """A tool phase still reserves a replica even with zero model requests."""
    controller.begin_stream_collection()
    controller._choose_worker(reserve_workflow=True)

    assert not controller.try_drain_stream_replica(0)
    controller.scheduler.async_call_engine.assert_not_called()


def test_stream_drain_idle_worker_targets_only_one_replica(controller):
    """Switching a replica must not broadcast pause or release to others."""
    controller.begin_stream_collection()

    assert controller.try_drain_stream_replica(1)

    calls = controller.scheduler.async_call_engine.call_args_list
    assert [call.kwargs["method"] for call in calls] == [
        "pause",
        "pause_generation",
        "offload",
    ]
    assert {call.kwargs["worker_id"] for call in calls} == {"rollout-1"}
    assert calls[-1].kwargs["tags"] == ["kv_cache", "weights", "cuda_graph"]
    assert controller._choose_worker()[1] == 0
    assert controller._choose_worker()[1] == 2


def test_stream_drain_last_replica_preserves_rejection_capacity(controller):
    """A final replica stays available for replacement groups after filtering."""
    controller.begin_stream_collection()
    for rank in range(3):
        assert controller.try_drain_stream_replica(rank)

    assert not controller.try_drain_stream_replica(3)
    assert controller._choose_worker()[1] == 3


def test_stream_drain_pair_active_member_keeps_both_replicas_open(controller):
    """A partial actor GPU set must not be fenced or offloaded."""
    controller.begin_stream_collection()
    controller._stream_active[1] = 1

    assert not controller.try_drain_stream_replicas([0, 1])

    assert not controller._stream_closed
    controller.scheduler.async_call_engine.assert_not_called()


def test_stream_drain_pair_fences_both_before_first_rpc(controller):
    """Dispatcher admission cannot slip between the two member checks."""
    controller.begin_stream_collection()

    async def verify_fence(**kwargs):
        assert {0, 1} <= controller._stream_closed
        assert controller._choose_worker()[1] in {2, 3}

    controller.scheduler.async_call_engine.side_effect = verify_fence

    assert controller.try_drain_stream_replicas([0, 1])
    assert controller.try_drain_stream_replicas([0, 1])

    assert controller._stream_drained == {0, 1}
    assert controller.scheduler.async_call_engine.call_count == 6
    assert not controller.try_drain_stream_replicas([2, 3])
    assert controller._stream_closed == {0, 1}


def test_stream_drain_pair_failure_keeps_entire_actor_gpu_set_fenced(controller):
    controller.begin_stream_collection()
    controller.scheduler.async_call_engine.side_effect = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        controller.try_drain_stream_replicas([0, 1])

    assert controller._stream_closed == {0, 1}
    assert not controller._stream_drained
    with pytest.raises(RuntimeError, match="transition failed"):
        controller._choose_worker()


def test_stream_finish_draining_releases_each_replica_once(controller):
    """A final barrier drains the last server without releasing others twice."""
    controller.begin_stream_collection()
    assert controller.try_drain_stream_replica(0)

    assert controller.finish_stream_draining() == [0, 1, 2, 3]
    assert controller.finish_stream_draining() == [0, 1, 2, 3]

    calls = controller.scheduler.async_call_engine.call_args_list
    assert len(calls) == 12
    with pytest.raises(RuntimeError, match="closed admission"):
        controller._choose_worker()


def test_stream_finish_draining_active_workflow_rejects_barrier(controller):
    """Joining generation is mandatory before closing the final replica."""
    controller.begin_stream_collection()
    controller._choose_worker(reserve_workflow=True)
    with pytest.raises(RuntimeError, match="workflows are active"):
        controller.finish_stream_draining()


def test_stream_drain_rpc_failure_fails_closed(controller):
    """A failed memory release never reopens the replica or reports success."""
    controller.begin_stream_collection()
    controller.scheduler.async_call_engine.side_effect = RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        controller.try_drain_stream_replica(1)

    assert 1 in controller._stream_closed
    with pytest.raises(RuntimeError, match="transition failed"):
        controller._choose_worker()
    with pytest.raises(RuntimeError, match="transition failed"):
        controller.get_stream_completed()


def test_stream_consume_selected_group_publishes_same_trajectory(controller):
    """Only consumed accepted groups are exposed, preserving tensor identity."""
    controller.begin_stream_collection()
    trajectory = {"example": object()}
    controller._stream_task_workers[11] = 2
    result = _RolloutResult(task_id=11, trajectory=trajectory, metrics={})

    assert controller._consume_rollout_result(result) is result
    (event,) = controller.get_stream_completed()

    assert event["task_id"] == 11
    assert event["worker_rank"] == 2
    assert event["trajectory"] is trajectory
    assert controller.get_stream_completed() == []


def test_stream_consume_rejected_group_does_not_publish(controller):
    """Filtering keeps the ordinary replacement path and emits no train work."""
    controller.begin_stream_collection()
    controller._stream_task_workers[11] = 2

    assert (
        controller._consume_rollout_result(
            _RolloutResult(task_id=11, trajectory=None, metrics={})
        )
        is None
    )
    assert controller.get_stream_completed() == []


def test_stream_restore_reopens_only_after_collection_ends(controller):
    """Admission restoration does not issue premature actor-memory onloads."""
    controller.begin_stream_collection()
    assert controller.try_drain_stream_replica(0)
    with pytest.raises(RuntimeError, match="during collection"):
        controller.restore_stream_replicas()
    controller.end_stream_collection()
    controller.scheduler.async_call_engine.reset_mock()

    controller.restore_stream_replicas()

    assert not controller._stream_closed
    controller.scheduler.async_call_engine.assert_not_called()


def test_stream_begin_backend_routing_rejects_unpinned_execution(controller):
    """Router leases could execute a workflow on another supposedly idle GPU."""
    controller._backend_router = object()
    with pytest.raises(ValueError, match="backend_routing"):
        controller.begin_stream_collection()


@pytest.mark.asyncio
async def test_stream_workflow_timeout_releases_admission_reservation(controller):
    """A callback timeout releases the whole-workflow reservation in finally."""
    controller.begin_stream_collection()
    controller._futures_lock = threading.Lock()
    controller._pending_futures = {}
    controller._staleness_manager = Mock()
    controller._callback_host = "localhost"
    controller._callback_port = 1234
    controller.config = SimpleNamespace(request_timeout=0.001)
    controller.scheduler.async_call_engine.return_value = 11
    task = _RemoteRolloutTaskInput(
        task_id=11,
        data={},
        workflow="example.Workflow",
        workflow_kwargs={},
        should_accept_fn=None,
    )

    assert await controller._create_submit_callback(task)() is None

    assert controller._stream_active[0] == 0
    assert not controller._pending_futures
    controller._staleness_manager.on_rollout_rejected.assert_called_once()
    with pytest.raises(RuntimeError, match="transition failed"):
        controller.try_drain_stream_replica(0)


def test_stream_topology_repeated_physical_gpu_rejects_overlap(controller):
    """Controller ranks must never be mistaken for proof of GPU placement."""
    controller._gpus_per_server = 1
    controller._collective_rpc = Mock(
        return_value=[
            {"host": "host", "gpu_uuids": ["GPU-a"]},
            {"host": "host", "gpu_uuids": ["GPU-a"]},
        ]
    )
    with pytest.raises(ValueError, match="overlap physical"):
        controller.stream_topology()


def test_stream_topology_reads_model_devices_instead_of_all_cuda_contexts():
    """An unrelated parent CUDA context is not a model TP device."""
    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.local_server_processes = [
        SimpleNamespace(host="host", process=SimpleNamespace(poll=lambda: None))
    ]
    engine.backend = SimpleNamespace(
        get_stream_topology=Mock(return_value=["GPU-first", "GPU-second"])
    )

    assert engine.get_stream_topology() == {
        "host": "host",
        "gpu_uuids": ["GPU-first", "GPU-second"],
    }
    engine.backend.get_stream_topology.assert_called_once_with()


@pytest.fixture
def proc_start_time(monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            return "10 (scheduler name) " + " ".join(["0"] * 20)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return "0"


def test_stream_topology_model_records_require_complete_live_tp(
    monkeypatch, tmp_path, proc_start_time
):
    import json

    from areal.engine.sglang_remote import SGLangBackend

    backend = SGLangBackend()
    backend._stream_topology_dir = SimpleNamespace(name=str(tmp_path))
    backend._stream_topology_size = 2
    alive = Mock()
    monkeypatch.setattr("areal.engine.sglang_remote.os.kill", alive)
    for rank in range(2):
        record = dict(
            pid=10 + rank,
            process_start_time=proc_start_time,
            tp_rank=rank,
            tp_size=2,
            gpu_uuid=f"GPU-{rank}",
        )
        (tmp_path / f"{10 + rank}.json").write_text(json.dumps(record))
    assert backend.get_stream_topology() == ["GPU-0", "GPU-1"]
    assert alive.call_count == 2
    (tmp_path / "11.json").unlink()
    with pytest.raises(RuntimeError, match="Incomplete SGLang model topology"):
        backend.get_stream_topology()


def test_stream_topology_publisher_reads_explicit_model_device(
    monkeypatch, tmp_path, proc_start_time
):
    import json

    import torch

    from areal.engine.awex.plugins.sglang_plugin import AwexSchedulerPlugin

    scheduler = SimpleNamespace(
        tp_rank=1,
        tp_size=2,
        tp_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    parameters=lambda: iter(
                        [SimpleNamespace(device=torch.device("cuda:3"))]
                    )
                )
            )
        ),
    )
    properties = Mock(return_value=SimpleNamespace(uuid="model-device-three"))
    monkeypatch.setenv("AREAL_SGLANG_STREAM_TOPOLOGY_DIR", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    AwexSchedulerPlugin(scheduler)._publish_stream_topology()
    properties.assert_called_once_with(torch.device("cuda:3"))
    records = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(records) == 1
    assert records[0]["gpu_uuid"] == "model-device-three"
    assert records[0]["tp_rank"] == 1
