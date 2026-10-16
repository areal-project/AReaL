# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from areal.infra.async_task_runner import TimedResult
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    WorkflowExecutor,
    _RolloutResult,
    _RolloutTaskInput,
)
from areal.utils import stats_tracker
from areal.utils.rollout_metrics import ConsumedMetrics, capture_metrics
from areal.utils.stats_tracker import DistributedStatsTracker


def dispatcher_with_results(results):
    buffer = ConsumedMetrics()
    dispatcher = BatchTaskDispatcher(16, Mock(), Mock(), consume_result=buffer.consume)
    for result in results:
        dispatcher._active_task_ids.add(result.task_id)
        dispatcher._pending_results[result.task_id] = TimedResult(
            create_time=result.task_id, task_id=result.task_id, data=result
        )
    return dispatcher, buffer


def test_prefetched_and_filtered_rewards_follow_consumed_attempts():
    """Exports omit prefetch, include rejected rewards, and reset exactly once."""
    tracker = DistributedStatsTracker("rollout")
    results = []
    for task_id, rewards in enumerate(([1.0, 0.0], [0.0], [0.9] * 3)):
        with capture_metrics() as metrics:
            for reward in rewards:
                tracker.scalar(reward=reward, harness_error=float(reward == 0))
        results.append(_RolloutResult(task_id, None if task_id == 1 else {}, metrics))
    dispatcher, buffer = dispatcher_with_results(results)
    assert tracker.export() == {}
    assert buffer.export() == {}
    selected = dispatcher.wait_results(2)
    assert sum(result is not None for result in selected) == 1
    stats = buffer.export()
    assert stats["rollout/reward"] == pytest.approx(1 / 3)
    assert stats["rollout/harness_error"] == pytest.approx(2 / 3)
    assert stats["rollout/reward__count"] == 3
    assert stats["rollout/reward_sample_count"] == 3
    assert buffer.export() == {}
    dispatcher.wait_results(1)
    assert buffer.export()["rollout/reward"] == pytest.approx(0.9)


@pytest.mark.parametrize("trajectory", [None, {"text": "ok"}])
def test_rpc_transfer_does_not_publish_worker_metrics(trajectory):
    """Both accepted and rejected sidecars survive serialization and prefetch."""
    result = _RolloutResult(0, trajectory, {"rollout/reward": [0.0, 1.0]})
    worker, worker_buffer = dispatcher_with_results([result])
    raw = worker.wait_for_task(0, consume=False)
    transported = deserialize_value(
        serialize_value({"trajectory": raw.trajectory, "metrics": raw.metrics})
    )
    assert worker_buffer.export() == {}
    controller, controller_buffer = dispatcher_with_results(
        [_RolloutResult(0, **transported)]
    )
    assert controller_buffer.export() == {}
    consumed = controller.wait_for_task(0)
    assert (consumed is None) == (trajectory is None)
    assert controller_buffer.export() == {
        "rollout/reward": 0.5,
        "rollout/reward__count": 2,
        "rollout/reward_sample_count": 2,
    }


@pytest.mark.asyncio
async def test_capture_isolates_concurrent_groups_and_inherits_into_children():
    """Async group children share only their parent's metrics context."""
    tracker = DistributedStatsTracker("rollout")

    async def child(reward):
        await asyncio.sleep(0)
        tracker.scalar(reward=reward)

    async def group(reward):
        with capture_metrics() as metrics:
            await asyncio.gather(child(reward), child(reward))
        return metrics

    first, second = await asyncio.gather(group(0), group(1))
    assert first == {"rollout/reward": [0.0, 0.0]}
    assert second == {"rollout/reward": [1.0, 1.0]}
    tracker.scalar(reward=0.25)
    assert tracker.export()["rollout/reward"] == 0.25


def test_capture_leaves_eval_operational_and_prm_metrics_in_export_window():
    """The batch buffer only intercepts training quality scalar metrics."""
    tracker = DistributedStatsTracker("rollout")
    evaluation = DistributedStatsTracker("eval-rollout")
    with capture_metrics() as metrics:
        tracker.scalar(
            **{
                "arena/call_success": 1,
                "prm_metric/rate": 0.5,
                "spec_accept_rate": 0.75,
                "reward": 0,
            }
        )
        evaluation.scalar(reward=1)
    assert metrics == {"rollout/reward": [0.0]}
    assert tracker.export()["rollout/arena/call_success"] == 1
    assert evaluation.export()["eval-rollout/reward"] == 1


def test_capture_exception_restores_context_and_preserves_observations():
    """A failing attempt retains already recorded metrics without leaking scope."""
    tracker = DistributedStatsTracker("rollout")
    with pytest.raises(ValueError), capture_metrics() as metrics:
        tracker.scalar(reward=0)
        raise ValueError("failed")
    assert metrics == {"rollout/reward": [0.0]}
    tracker.scalar(reward=1)
    assert tracker.export()["rollout/reward"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["accepted", "filtered", "none", "exception"])
async def test_workflow_quality_survives_rejection_and_failure(outcome, monkeypatch):
    """The real workflow wrapper retains observations for every completed attempt."""
    tracker = DistributedStatsTracker("rollout")
    monkeypatch.setattr(stats_tracker, "get", lambda name: tracker)

    class Workflow:
        async def arun_episode(self, engine, data):
            tracker.scalar(reward=0.75, harness_error=0)
            if outcome == "exception":
                raise ValueError("episode failed")
            return None if outcome == "none" else {"text": "ok"}

    executor = WorkflowExecutor.__new__(WorkflowExecutor)
    executor.config = SimpleNamespace(
        check_trajectory_format=False, dump_to_file=False, enable_rollout_tracing=False
    )
    executor.inference_engine = Mock()
    executor._staleness_manager = Mock()
    executor.logger = Mock()
    task = _RolloutTaskInput(0, {}, Workflow(), lambda traj: outcome != "filtered")
    result = await executor._create_workflow_task(task)()
    assert tracker.export() == {}
    assert result.metrics["rollout/reward"] == [0.75]
    accepted = outcome == "accepted"
    assert (result.trajectory is not None) == accepted
    buffer = ConsumedMetrics()
    buffer.consume(result)
    stats = buffer.export()
    assert stats["rollout/group_attempt_count"] == 1
    assert stats["rollout/accepted"] == float(accepted)
    assert stats["rollout/rejected"] == float(not accepted)


def test_rpc_envelope_preserves_remote_tensor_transport(monkeypatch):
    """Real RPC envelopes allow tensor remotizing before serialization."""
    backend = Mock()
    backend.store.return_value = "test-shard"
    monkeypatch.setattr("areal.infra.rpc.rtensor.get_backend", lambda: backend)
    tensor = torch.ones((1, 3), dtype=torch.long)
    result = _RolloutResult(0, {"input_ids": tensor}, {"rollout/reward": [1.0]})
    worker, buffer = dispatcher_with_results([result])
    engine = SimpleNamespace(
        workflow_executor=SimpleNamespace(
            _wait_for_task_result=lambda *args: worker.wait_for_task(
                *args, consume=False
            )
        )
    )
    payload = RemoteInfEngine._wait_for_task_result(engine, 0)
    remote = RTensor.remotize(payload, node_addr="localhost:1234")
    restored = deserialize_value(serialize_value(remote))
    assert isinstance(restored["trajectory"]["input_ids"], RTensor)
    assert restored["metrics"] == result.metrics
    assert buffer.export() == {}
    backend.store.assert_called_once()


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_engine_facade_transfers_metrics_and_exports_only_consumed(
    backend, monkeypatch
):
    """Public RPC targets forward sidecars and expose local consumed metrics."""
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.engine.vllm_remote import RemotevLLMEngine

    engine_cls = RemoteSGLangEngine if backend == "sglang" else RemotevLLMEngine
    engine = engine_cls.__new__(engine_cls)
    buffer = ConsumedMetrics()
    inner = Mock()
    inner._workflow_executor = SimpleNamespace(consumed_metrics=buffer)
    engine._engine = inner
    monkeypatch.setattr(stats_tracker, "export_all", lambda **kwargs: {"system": 2})
    engine._wait_for_task_result(7, timeout=1, raise_timeout=False)
    inner._wait_for_task_result.assert_called_once_with(7, 1, False)
    buffer.consume(_RolloutResult(0, None, {"rollout/reward": [1, 0, 0]}))
    result = engine.export_stats()
    assert result["rollout/reward"] == pytest.approx(1 / 3)
    assert result["rollout/reward__count"] == 3
    assert result["system"] == 2
    assert engine.export_stats() == {"system": 2}


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_engine_stats_export_before_initialization_remains_supported(
    backend, monkeypatch
):
    """Global operational stats remain exportable before workflow initialization."""
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.engine.vllm_remote import RemotevLLMEngine

    engine_cls = RemoteSGLangEngine if backend == "sglang" else RemotevLLMEngine
    engine = engine_cls.__new__(engine_cls)
    engine._engine = SimpleNamespace(_workflow_executor=None)
    monkeypatch.setattr(stats_tracker, "export_all", lambda **kwargs: {"system": 2})
    assert engine.export_stats() == {"system": 2}


def test_terminal_failure_is_transported_then_counted_once():
    from areal.infra.workflow_executor import (
        WorkflowContractError,
        WorkflowContractFailure,
        get_workflow_result_error,
    )

    failure = WorkflowContractFailure("invalid group", {"rollout/accepted": [0.0]})
    worker = BatchTaskDispatcher(
        16, Mock(), Mock(), terminal_error_fn=get_workflow_result_error
    )
    worker._active_task_ids.add(1)
    worker._pending_results[1] = TimedResult(create_time=1, task_id=1, data=failure)
    transported = worker.wait_for_task(1, raise_terminal_error=False, consume=False)
    consumer = ConsumedMetrics()
    controller = BatchTaskDispatcher(
        16,
        Mock(),
        Mock(),
        consume_result=consumer.consume,
        terminal_error_fn=get_workflow_result_error,
    )
    controller._active_task_ids.add(1)
    controller._pending_results[1] = TimedResult(
        create_time=1, task_id=1, data=transported
    )
    with pytest.raises(WorkflowContractError, match="invalid group"):
        controller.wait_results(1)
    assert consumer.export()["rollout/group_attempt_count"] == 1
    assert consumer.export() == {}


def test_v2_exports_consumed_quality_without_waiting_for_new_completion(monkeypatch):
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )

    controller = RolloutControllerV2.__new__(RolloutControllerV2)
    buffer = ConsumedMetrics()
    controller._workflow_executor = SimpleNamespace(consumed_metrics=buffer)
    monkeypatch.setattr(stats_tracker, "export_all", lambda **kwargs: {})
    buffer.consume(
        _RolloutResult(1, None, {"rollout/reward": [1, 0], "rollout/accepted": [0]})
    )
    stats = controller.export_stats()
    assert stats["rollout/reward"] == 0.5
    assert stats["rollout/reward_sample_count"] == 2
    assert stats["rollout/group_attempt_count"] == 1
    assert controller.export_stats() == {}
