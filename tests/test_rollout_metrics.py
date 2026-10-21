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
    _RolloutResult,
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


def test_rpc_envelope_preserves_remote_tensor_transport(monkeypatch):
    """Real RPC envelopes allow tensor remotizing before serialization."""
    backend = Mock()
    backend.store.return_value = "test-shard"
    monkeypatch.setattr("areal.infra.rpc.rtensor.get_backend", lambda: backend)
    tensor = torch.ones((1, 3), dtype=torch.long)
    result = _RolloutResult(0, {"input_ids": tensor}, {"rollout/reward": [1.0]})
    worker, buffer = dispatcher_with_results([result])
    engine = SimpleNamespace(workflow_executor=SimpleNamespace(dispatcher=worker))
    payload = RemoteInfEngine.get_rollout_result(engine, 0)
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
    engine.get_rollout_result(7, timeout=1, raise_timeout=False)
    inner.get_rollout_result.assert_called_once_with(7, 1, False)
    buffer.consume(_RolloutResult(0, None, {"rollout/reward": [1, 0, 0]}))
    result = engine.export_stats()
    assert result["rollout/reward"] == pytest.approx(1 / 3)
    assert result["rollout/reward__count"] == 3
    assert result["system"] == 2
    assert engine.export_stats() == {"system": 2}
