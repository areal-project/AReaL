# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.test_workflow_executor_filtering import _executor, _TrajectoryWorkflow

from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.workflow_executor import _RolloutTaskInput
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.stats import log_arena_stream_training_counts
from areal.utils import stats_tracker
from areal.utils.data import RolloutGroup, TrajBatchMeta, concat_batch, split_batch
from areal.utils.rollout_metrics import ConsumedMetrics, get_arena_stream_name


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
        "loss_mask": torch.tensor([False, True, True, False]).repeat(rows, 1),
        "rollout_group": RolloutGroup(counts, arena_stream_name=stream),
    }


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (None, None),
        ({"dataset": "ordinary"}, None),
        ({"arena_stream_name": "swe", "stream_id": "remote-id"}, "swe"),
        ({"stream_id": "remote-id"}, "remote-id"),
        ({"arena_stream_name": "", "stream_id": "remote-id"}, "remote-id"),
    ],
)
def test_arena_stream_name_uses_configured_name_then_id(data, expected):
    assert get_arena_stream_name(data) == expected


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
            assert result.trajectory["rollout_group"].arena_stream_name == stream

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
def test_arena_stream_metadata_survives_rpc_concat_and_split(stream):
    """Arena stream provenance stays outside model tensors across transport boundaries."""
    trajectory = deserialize_value(serialize_value(_trajectory(stream)))
    batched, meta = concat_batch([trajectory])
    assert "rollout_group" not in batched
    restored = split_batch(batched, meta)
    assert restored[0]["rollout_group"] == RolloutGroup(
        (2, 1), arena_stream_name=stream
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("correct_count", [1, 2])
async def test_accuracy_predicate_rejections_reach_arena_stream_counts(correct_count):
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
    """Count input denominators once, after update success."""
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
    assert stats["ppo_actor/stream/a/n_groups"] == 2
    assert stats["ppo_actor/stream/a/n_seqs"] == 4
    assert stats["ppo_actor/stream/a/n_valid_tokens"] == 8
    assert stats["ppo_actor/stream/b/n_groups"] == 1
    assert stats["ppo_actor/stream/b/n_seqs"] == 3
    assert stats["ppo_actor/stream/b/n_valid_tokens"] == 6
    assert stats["ppo_actor/stream/a/n_tokens"] == 16
    assert stats["ppo_actor/stream/b/n_tokens"] == 12
    assert not any(
        key.endswith(("/groups", "/rollouts", "/tokens", "/sequences")) for key in stats
    )


def test_actor_token_counts_exclude_masked_and_unnamed_rows_after_padding():
    actor = object.__new__(PPOActor)
    actor._ppo_update = MagicMock()
    short = _trajectory("a", (2,))
    for key in ("input_ids", "attention_mask", "loss_mask"):
        short[key] = short[key][:, :2]
    unnamed = _trajectory(None, (3,))
    unnamed.pop("rollout_group")
    masked = _trajectory("b", (3,))
    masked["loss_mask"].zero_()
    repeated = _trajectory("a", (1,))
    data = [short, unnamed, masked, repeated]
    for trajectory in data:
        trajectory["advantages"] = torch.zeros_like(
            trajectory["loss_mask"], dtype=torch.float32
        )

    actor.ppo_update(data)

    stats = stats_tracker.export_all()
    assert stats["ppo_actor/stream/a/n_valid_tokens"] == 4
    assert stats["ppo_actor/stream/a/n_groups"] == 2
    assert stats["ppo_actor/stream/a/n_seqs"] == 3
    assert stats["ppo_actor/stream/b/n_valid_tokens"] == 0
    assert stats["ppo_actor/stream/b/n_groups"] == 1
    assert stats["ppo_actor/stream/b/n_seqs"] == 3
    assert stats["ppo_actor/stream/a/n_tokens"] == 12
    assert stats["ppo_actor/stream/b/n_tokens"] == 12
    assert not any("None" in key for key in stats)


def _arena_stream_counts_worker(rank, init_method):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=init_method,
        timeout=timedelta(seconds=60),
    )
    try:
        # Both DP ranks have a; only rank 0 has b. Keys must still synchronize.
        groups = [RolloutGroup((2, 1), arena_stream_name="a")]
        if rank == 0:
            groups.append(RolloutGroup((2,), arena_stream_name="b"))
        meta = TrajBatchMeta(
            len(groups), [sum(g.row_counts) for g in groups], [4] * len(groups), groups
        )
        loss_mask = torch.tensor([False, True, True, False]).repeat(
            sum(meta.traj_group_sizes), 1
        )
        with stats_tracker.scope("ppo_actor"):
            log_arena_stream_training_counts(meta, loss_mask)
        stats = stats_tracker.export_all(reduce_group=dist.group.WORLD)
        assert stats["ppo_actor/stream/a/n_groups"] == 2
        assert stats["ppo_actor/stream/a/n_seqs"] == 6
        assert stats["ppo_actor/stream/a/n_tokens"] == 24
        assert stats["ppo_actor/stream/b/n_seqs"] == 2
        assert stats["ppo_actor/stream/b/n_tokens"] == 8
        assert stats["ppo_actor/stream/a/n_valid_tokens"] == 12
        assert stats["ppo_actor/stream/b/n_groups"] == 1
        assert stats["ppo_actor/stream/b/n_valid_tokens"] == 4

    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_arena_stream_training_counts_sum_across_dp_with_missing_stream(tmp_path):
    mp.spawn(
        _arena_stream_counts_worker,
        args=(f"file://{tmp_path / 'init'}",),
        nprocs=2,
        join=True,
    )


def test_stream_counts_sum_to_global_ppo_denominators():
    from areal.trainer.ppo.actor import _group_training_metrics
    from areal.trainer.ppo.stats import infer_token_denominator

    short = _trajectory("a", (2, 1))
    for key in ("input_ids", "attention_mask", "loss_mask"):
        short[key] = short[key][:, :2]
    masked = _trajectory("b", (2,))
    masked["loss_mask"].zero_()
    batched, meta = concat_batch([short, masked, _trajectory("a", (1,))])
    mask = batched["loss_mask"]
    with stats_tracker.scope("ppo_actor"):
        stats_tracker.denominator(
            n_groups=_group_training_metrics(mask, meta.traj_group_sizes)[0],
            n_seqs=torch.ones(mask.shape[0], dtype=torch.bool),
            n_tokens=infer_token_denominator(batched, mask),
            n_valid_tokens=mask.bool(),
        )
        log_arena_stream_training_counts(meta, mask)
    stats = stats_tracker.export_all()
    for name in ("n_groups", "n_seqs", "n_tokens", "n_valid_tokens"):
        assert (
            sum(stats[f"ppo_actor/stream/{stream}/{name}"] for stream in ("a", "b"))
            == stats[f"ppo_actor/{name}"]
        )
    assert stats["ppo_actor/n_tokens"] == 24
    assert stats["ppo_actor/n_valid_tokens"] == 5
