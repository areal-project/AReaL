# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_workflow_executor_filtering import _executor, _TrajectoryWorkflow

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.workflow_executor import _RolloutTaskInput
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.stats import log_stream_training_counts
from areal.utils import stats_tracker
from areal.utils.data import RolloutGroup, TrajBatchMeta, concat_batch, split_batch
from areal.utils.rollout_metrics import ConsumedMetrics, reduce_rollout_metrics


@pytest.fixture(autouse=True)
def isolated_stats(monkeypatch):
    stats_tracker.DEFAULT_TRACKER.export()
    monkeypatch.setattr(stats_tracker, "TRACKERS", {"": stats_tracker.DEFAULT_TRACKER})
    yield
    stats_tracker.DEFAULT_TRACKER.export()


def _trajectory(stream="a", counts=(2, 1)):
    rows = sum(counts)
    return {
        "input_ids": torch.ones((rows, 4), dtype=torch.long),
        "attention_mask": torch.ones((rows, 4), dtype=torch.bool),
        "rollout_group": RolloutGroup(counts, stream_name=stream),
    }


@pytest.mark.asyncio
async def test_filter_counts_only_explicit_rejections_and_consumed_results():
    """None/error results are rejected, but only predicate false is filtered."""
    executor, _ = _executor()
    consumed = ConsumedMetrics()

    def raises(_traj):
        raise RuntimeError("predicate failed")

    cases = [
        ("a", _trajectory(), lambda _: True),
        ("a", _trajectory(), lambda _: False),
        ("a", None, lambda _: False),
        ("b", _trajectory(), raises),
        ("b", _trajectory(), None),
    ]
    for index, (stream, traj, predicate) in enumerate(cases):
        result = await executor._create_workflow_task(
            _RolloutTaskInput(
                task_id=index,
                data={"arena_stream_name": stream},
                workflow=_TrajectoryWorkflow(traj),
                should_accept_fn=predicate,
            )
        )()
        consumed.consume(result)
        if result.trajectory is not None:
            assert result.trajectory["rollout_group"].stream_name == stream

    # An accepted prefetched result has not yet entered the consumer's window.
    await executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=99,
            data={"arena_stream_name": "pending"},
            workflow=_TrajectoryWorkflow(_trajectory()),
        )
    )()
    stats = consumed.export()
    assert stats["rollout/stream/a/group_attempt_count"] == 3
    assert stats["rollout/stream/a/accepted_count"] == 1
    assert stats["rollout/stream/a/rejected_count"] == 2
    assert stats["rollout/stream/a/filter_rejected_count"] == 1
    assert stats["rollout/stream/b/group_attempt_count"] == 2
    assert stats["rollout/stream/b/rejected_count"] == 1
    assert stats["rollout/stream/b/filter_rejected_count"] == 0
    assert stats["rollout/filter_rejected_count"] == 1
    assert not any("pending" in key for key in stats)
    assert consumed.export() == {}


@pytest.mark.parametrize("stream", ["a", "org/dataset", None])
def test_stream_metadata_survives_rpc_concat_and_split(stream):
    """Stream provenance stays outside model tensors across transport boundaries."""
    trajectory = deserialize_value(serialize_value(_trajectory(stream)))
    batched, meta = concat_batch([trajectory])
    assert "rollout_group" not in batched
    restored = split_batch(batched, meta)
    assert restored[0]["rollout_group"] == RolloutGroup((2, 1), stream_name=stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("correct_count", [1, 2])
async def test_accuracy_predicate_rejections_reach_stream_counts(correct_count):
    from examples.swe.filter_function import filter_by_accuracy

    executor, _ = _executor()
    trajectory = _trajectory(counts=(1, 1, 1, 1, 1))
    trajectory["original_rewards"] = torch.tensor(
        [1.0] * correct_count + [0.0] * (5 - correct_count)
    )
    result = await executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=1,
            data={"arena_stream_name": "swe"},
            workflow=_TrajectoryWorkflow(trajectory),
            should_accept_fn=filter_by_accuracy,
        )
    )()
    consumed = ConsumedMetrics()
    consumed.consume(result)
    stats = consumed.export()
    assert (result.trajectory is not None) == (correct_count > 1)
    assert stats["rollout/stream/swe/filter_rejected_count"] == int(correct_count == 1)
    assert stats["rollout/stream/swe/filter_accuracy"] == correct_count / 5


@pytest.mark.parametrize("fails", [False, True])
def test_actor_counts_successful_training_inputs_once(fails):
    """Count groups, logical rollouts and rows separately, after update success."""
    actor = object.__new__(PPOActor)
    actor._ppo_update = MagicMock(
        side_effect=RuntimeError("update failed") if fails else None
    )
    data = [_trajectory("a"), _trajectory("a", (1,)), _trajectory("b", (3,))]
    if fails:
        with pytest.raises(RuntimeError, match="update failed"):
            actor.ppo_update(data)
        assert stats_tracker.export_all() == {}
        return
    actor.ppo_update(data)
    stats = stats_tracker.export_all()
    assert stats["ppo_actor/stream/a/groups"] == 2
    assert stats["ppo_actor/stream/a/rollouts"] == 3
    assert stats["ppo_actor/stream/a/sequences"] == 4
    assert stats["ppo_actor/stream/b/groups"] == 1
    assert stats["ppo_actor/stream/b/rollouts"] == 1
    assert stats["ppo_actor/stream/b/sequences"] == 3


def _stream_counts_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=60),
    )
    try:
        # Both DP ranks have a; only rank 0 has b. Keys must still synchronize.
        groups = [RolloutGroup((2, 1), stream_name="a")]
        if rank == 0:
            groups.append(RolloutGroup((2,), stream_name="b"))
        meta = TrajBatchMeta(
            len(groups), [sum(g.row_counts) for g in groups], [4] * len(groups), groups
        )
        with stats_tracker.scope("ppo_actor"):
            log_stream_training_counts(meta)
        stats = stats_tracker.export_all(reduce_group=dist.group.WORLD)
        assert stats["ppo_actor/stream/a/groups"] == 2
        assert stats["ppo_actor/stream/a/rollouts"] == 4
        assert stats["ppo_actor/stream/a/sequences"] == 6
        assert stats["ppo_actor/stream/b/groups"] == 1
        assert stats["ppo_actor/stream/b/sequences"] == 2

        consumed = ConsumedMetrics()
        accepted = [1.0] if rank == 0 else [0.0, 0.0, 1.0]
        consumed.consume(
            SimpleNamespace(
                trajectory=None,
                metrics={
                    "rollout/stream/a/accepted": accepted,
                    "rollout/stream/a/filter_rejected": [1 - x for x in accepted],
                    "rollout/stream/a/harness_success": accepted,
                    "rollout/stream/a/task_error": [1 - x for x in accepted],
                },
            )
        )
        local_stats = consumed.export()
        local_stats["rollout/arena/latency"] = rank + 1
        local_stats["rollout/arena/latency__count"] = 1
        merged = reduce_rollout_metrics(
            local_stats, group=dist.group.WORLD, is_data_parallel_head=True
        )
        assert merged["rollout/stream/a/accepted"] == 0.5
        assert merged["rollout/stream/a/group_attempt_count"] == 4
        assert merged["rollout/stream/a/task_result_count"] == 4
        assert merged["rollout/stream/a/task_error"] == 0.5
        assert merged["rollout/stream/a/filter_rejected_count"] == 2
        assert merged["rollout/arena/latency"] == rank + 1

        # A model-parallel replica must not add observations to the DP head.
        merged = reduce_rollout_metrics(
            local_stats, group=dist.group.WORLD, is_data_parallel_head=rank == 0
        )
        assert merged["rollout/stream/a/accepted"] == 1
        assert merged["rollout/stream/a/group_attempt_count"] == 1
        assert merged["rollout/stream/a/filter_rejected_count"] == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_stream_training_counts_sum_across_dp_with_missing_stream(tmp_path):
    mp.spawn(
        _stream_counts_worker,
        args=(f"file://{tmp_path / 'init'}",),
        nprocs=2,
        join=True,
    )
