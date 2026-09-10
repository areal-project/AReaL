# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from areal.infra.controller.train_controller import TrainController
from areal.infra.rpc.rtensor import RTensor, RTensorDrainReceipt, TensorShardInfo
from areal.trainer.mopd.targets import (
    MOPD_CONTRIBUTIONS_KEY,
    aggregate_mopd_targets,
)


def _rtensor(shard_id: str, node_addr: str = "teacher:8000") -> RTensor:
    return RTensor(
        shard=TensorShardInfo(shard_id=shard_id, node_addr=node_addr),
        data=torch.empty(3, device="meta"),
    )


def test_megatron_backend_exposes_mopd_rpc_methods():
    """The supported MOPD backend exposes the controller's RPC surface."""
    import importlib

    actor_cls = getattr(
        importlib.import_module("areal.engine.megatron_engine"), "MegatronPPOActor"
    )

    assert callable(getattr(actor_cls, "aggregate_mopd_targets", None))


def test_aggregate_mopd_targets_uses_raw_weights_and_removes_teacher_metadata():
    """Actor aggregation preserves raw scale and drops route/contribution keys."""
    batch = [
        {
            "mopd_route": "ensemble",
            MOPD_CONTRIBUTIONS_KEY: {
                "teacher-a": {"logp": torch.tensor([1.0, 2.0]), "weight": 0.5},
                "teacher-b": {"logp": torch.tensor([3.0, 4.0]), "weight": 1.5},
            },
        },
        {
            "mopd_route": "single",
            MOPD_CONTRIBUTIONS_KEY: {
                "teacher-b": {"logp": torch.tensor([5.0, 6.0]), "weight": 2.0}
            },
        },
    ]

    result = aggregate_mopd_targets(batch)

    assert result is batch
    torch.testing.assert_close(
        batch[0]["mopd_teacher_logp_sum"],
        torch.tensor([5.0, 7.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        batch[0]["mopd_teacher_weight_sum"],
        torch.tensor([2.0, 2.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        batch[1]["mopd_teacher_weight_sum"],
        torch.tensor([2.0, 2.0]),
        rtol=0.0,
        atol=0.0,
    )
    for trajectory in batch:
        assert "mopd_route" not in trajectory
        assert MOPD_CONTRIBUTIONS_KEY not in trajectory
        assert set(trajectory) >= {
            "mopd_teacher_logp_sum",
            "mopd_teacher_weight_sum",
        }
        assert not any(key.endswith("coefficient") for key in trajectory)
        assert "mopd_importance_ratio_cap" not in trajectory


def test_aggregate_mopd_targets_rejects_mismatched_teacher_shapes():
    """All teachers contributing to one trajectory must use one token shape."""
    batch = [
        {
            "mopd_route": "bad",
            MOPD_CONTRIBUTIONS_KEY: {
                "teacher-a": {"logp": torch.ones(2), "weight": 1.0},
                "teacher-b": {"logp": torch.ones(3), "weight": 1.0},
            },
        }
    ]

    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_mopd_targets(batch)


def _strict_controller(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stats: list[dict[str, int]],
) -> tuple[TrainController, list[tuple[str, tuple[Any, ...]]]]:
    controller = object.__new__(TrainController)
    controller.workers_is_dp_head = [True, False, True]
    controller._worker_role = "actor"
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def call_all(method: str, *args: Any, **_: Any) -> list[Any]:
        calls.append((method, args))
        if method == "clear_batches":
            return [1, 1]
        if method == "fetch_buffer_stats":
            return stats
        raise AssertionError(method)

    monkeypatch.setattr(controller, "_custom_function_call_all_dp_heads", call_all)
    return controller, calls


def test_strict_clear_batches_covers_sources_and_every_actor_dp_head(monkeypatch):
    """A receipt is complete only after source and all actor heads are clean."""
    controller, calls = _strict_controller(
        monkeypatch,
        stats=[
            {"num_entries": 4, "matching_entries": 0},
            {"num_entries": 1, "matching_entries": 0},
        ],
    )
    source_calls: list[tuple[str, list[str]]] = []

    async def clear_node(node_addr: str, shard_ids: list[str]) -> int:
        source_calls.append((node_addr, shard_ids))
        return len(shard_ids)

    monkeypatch.setattr(RTensor, "clear_node", clear_node)
    targets = [
        {"logp": _rtensor("a")},
        {"logp": _rtensor("a")},
        {"logp": _rtensor("b", "teacher:8001")},
    ]

    receipt = controller.strict_clear_batches(targets)

    assert receipt == RTensorDrainReceipt(
        consumer_role="actor",
        shard_count=2,
        source_node_count=2,
        consumer_dp_head_count=2,
    )
    assert source_calls == [
        ("teacher:8000", ["a"]),
        ("teacher:8001", ["b"]),
    ]
    assert [method for method, _ in calls] == [
        "clear_batches",
        "fetch_buffer_stats",
    ]
    assert calls[0][1] == (["a", "b"],)


def test_rtensor_drain_receipt_is_frozen_and_role_typed():
    receipt = RTensorDrainReceipt(
        consumer_role="actor",
        shard_count=1,
        source_node_count=1,
        consumer_dp_head_count=2,
    )

    with pytest.raises(FrozenInstanceError):
        receipt.consumer_role = "teacher"


def test_strict_clear_batches_source_failure_prevents_receipt(monkeypatch):
    """A failed teacher source DELETE is fatal and actor clearing does not hide it."""
    controller, calls = _strict_controller(
        monkeypatch, stats=[{"matching_entries": 0}, {"matching_entries": 0}]
    )

    async def clear_node(_: str, __: list[str]) -> int:
        raise RuntimeError("source delete failed")

    monkeypatch.setattr(RTensor, "clear_node", clear_node)

    with pytest.raises(RuntimeError, match="source delete failed"):
        controller.strict_clear_batches({"logp": _rtensor("a")})

    assert calls == []


def test_strict_clear_batches_rejects_one_leaking_actor_head(monkeypatch):
    """Checking head zero is insufficient when another actor DP head still leaks."""
    controller, _ = _strict_controller(
        monkeypatch,
        stats=[
            {"num_entries": 0, "matching_entries": 0},
            {"num_entries": 1, "matching_entries": 1},
        ],
    )

    async def clear_node(_: str, shard_ids: list[str]) -> int:
        return len(shard_ids)

    monkeypatch.setattr(RTensor, "clear_node", clear_node)

    with pytest.raises(RuntimeError, match=r"DP heads \[1\]"):
        controller.strict_clear_batches({"logp": _rtensor("a")})
