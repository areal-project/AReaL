# SPDX-License-Identifier: Apache-2.0

"""Check targeted actor RPC dispatch without a distributed training runtime."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from areal.infra.controller.train_controller import TrainController


@pytest.fixture
def controller():
    """Two TP2 replicas; only the selected replica may participate in backward."""
    value = TrainController.__new__(TrainController)
    value.workers = [
        SimpleNamespace(id=f"actor-{rank}", ip="node") for rank in range(4)
    ]
    value.workers_is_dp_head = [True, False, True, False]
    value._stream_replica_groups = {frozenset({0, 1}), frozenset({2, 3})}
    value._worker_role = "actor"
    value.config = SimpleNamespace(request_timeout=37.0)
    value.rollout = SimpleNamespace(config=SimpleNamespace(mask_stale_tokens=False))
    value.scheduler = SimpleNamespace(
        async_call_engine=AsyncMock(return_value={"ok": True})
    )
    return value


def test_stream_accumulate_targets_selected_tp_replica_and_head_payload(controller):
    """Only ranks 2/3 communicate, and rank 2 supplies the shared group payload."""
    group = {"sample": object()}

    result = controller.stream_accumulate("update", 10, [group], worker_ranks=[2, 3])

    calls = controller.scheduler.async_call_engine.call_args_list
    assert len(calls) == 2
    by_worker = {call.args[0]: call for call in calls}
    assert set(by_worker) == {"actor-2", "actor-3"}
    head, follower = by_worker["actor-2"], by_worker["actor-3"]
    assert head.args[:3] == ("actor-2", "stream_accumulate", "actor/2")
    assert head.args[3:5] == ("update", 10)
    assert head.args[5][0] is group
    assert follower.args == ("actor-3", "stream_accumulate", "actor/3")
    assert head.kwargs["max_head_offpolicyness"] is None
    assert "max_head_offpolicyness" not in follower.kwargs
    for call in calls:
        assert call.kwargs["rpc_meta"] == {"broadcast": True}
        assert call.kwargs["max_retries"] == 1
        assert call.kwargs["http_timeout"] == 37.0
    assert result == {"ok": True}


def test_stream_worker_call_failure_reports_rank_after_other_ranks_finish(controller):
    """A failed rank cannot be mistaken for a successful streamed gradient."""
    completed = []

    async def execute(worker_id, method, engine_name, *args, **kwargs):
        completed.append(worker_id)
        if worker_id == "actor-3":
            raise RuntimeError("injected rank failure")
        return {"ok": True}

    controller.scheduler.async_call_engine.side_effect = execute
    with pytest.raises(
        ExceptionGroup, match="Stream actor RPC stream_accumulate failed"
    ) as caught:
        controller.stream_accumulate("update", 10, [{}], worker_ranks=[2, 3])

    assert set(completed) == {"actor-2", "actor-3"}
    assert "actor rank 3" in str(caught.value.exceptions[0])
    assert "injected rank failure" in str(caught.value.exceptions[0])


@pytest.mark.parametrize("ranks", [[2], [0, 3]])
def test_stream_accumulate_incomplete_or_mixed_tp_replica_rejected(controller, ranks):
    """A lone head or mixed replica cannot enter a TP payload broadcast."""
    with pytest.raises(ValueError, match="discovered complete actor replica"):
        controller.stream_accumulate("update", 10, [{}], worker_ranks=ranks)
    controller.scheduler.async_call_engine.assert_not_called()


def test_stream_topology_rpc_avoids_broadcast_and_uses_scheduler_hosts(controller):
    """Discovery must not broadcast onto a TP group that still runs inference."""

    async def execute(worker_id, method, engine_name, *args, **kwargs):
        rank = int(worker_id.rsplit("-", 1)[1])
        if not controller.workers_is_dp_head[rank]:
            return None  # The actual guard discards initialized non-head replies.
        members = [0, 1] if rank < 2 else [2, 3]
        return [
            {
                "rank": member,
                "host": "untrusted-worker-name",
                "replica_ranks": members,
            }
            for member in members
        ]

    controller.scheduler.async_call_engine.side_effect = execute

    records = controller.stream_topology()

    assert [record["host"] for record in records] == ["node"] * 4
    assert [record["rank"] for record in records] == list(range(4))
    for call in controller.scheduler.async_call_engine.call_args_list:
        assert call.kwargs["rpc_meta"] == {"broadcast": False}
        assert call.kwargs["max_retries"] == 1
