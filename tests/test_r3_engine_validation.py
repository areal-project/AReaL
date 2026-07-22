# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_engine import MegatronEngine
from areal.engine.r3.asserts import R3Error
from areal.engine.r3.config import R3MoEConfig


def _fake_engine(
    routed_experts: torch.Tensor,
    *,
    routing_valid: torch.Tensor | None = None,
    num_moe_layers: int = 2,
    topk: int = 2,
    num_moe_experts: int = 4,
) -> MegatronEngine:
    engine = object.__new__(MegatronEngine)
    engine._r3_enabled = True
    engine.enable_tree_training = False
    engine.is_vision_model = False
    engine.use_padded_seq = False
    engine.device = torch.device("cpu")
    engine._r3_pending_routed_experts = routed_experts
    engine._r3_pending_valid = (
        torch.ones(
            routed_experts.shape[0],
            dtype=torch.bool,
        )
        if routing_valid is None
        else routing_valid
    )
    engine._r3_moe_config = R3MoEConfig(
        num_layers=num_moe_layers,
        num_moe_layers=num_moe_layers,
        topk=topk,
        moe_layer_indices=tuple(range(num_moe_layers)),
        global_to_moe_index={idx: idx for idx in range(num_moe_layers)},
    )
    engine._r3_num_moe_experts = num_moe_experts
    return engine


def _fake_mb_list(batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        forward_indices=None,
        mbs=[{"cu_seqlens": torch.tensor([0, batch_size], dtype=torch.int32)}],
        group_lens=[batch_size],
        padding_lengths=[0],
        padded_to_lengths=[batch_size],
    )


def test_prepare_r3_forward_context_rejects_layer_topk_mismatch():
    routed = torch.ones(1, 4, 3, 2, dtype=torch.int32)
    engine = _fake_engine(routed, num_moe_layers=2, topk=2)

    with pytest.raises(R3Error, match="layer/topk dims do not match"):
        engine._prepare_r3_forward_context(_fake_mb_list(batch_size=1))


@pytest.mark.parametrize("bad_expert_id", [-1, 4])
def test_prepare_r3_forward_context_rejects_expert_id_out_of_range(
    bad_expert_id: int,
):
    routed = torch.ones(1, 4, 2, 2, dtype=torch.int32)
    routed[0, 0, 0, 0] = bad_expert_id
    engine = _fake_engine(routed, num_moe_experts=4)

    with pytest.raises(R3Error, match="outside Megatron MoE range"):
        engine._prepare_r3_forward_context(_fake_mb_list(batch_size=1))


def test_prepare_r3_forward_context_records_batch_validity_stats():
    routed = torch.ones(3, 4, 2, 2, dtype=torch.int32)
    valid = torch.tensor([True, False, True])
    engine = _fake_engine(routed, routing_valid=valid)
    mb_list = SimpleNamespace(
        forward_indices=[2, 0, 1],
        mbs=[
            {"cu_seqlens": torch.tensor([0, 4, 8], dtype=torch.int32)},
            {"cu_seqlens": torch.tensor([0, 4], dtype=torch.int32)},
        ],
        group_lens=[8, 4],
        padding_lengths=[2, 0],
        padded_to_lengths=[10, 4],
    )

    context = engine._prepare_r3_forward_context(mb_list)

    assert context["valid_mbs"][0].tolist() == [True, True]
    assert context["valid_mbs"][1].tolist() == [False]
    assert context["real_tokens_by_mb"] == [8, 4]
    stats = context["stats"]
    assert stats["batches_with_side_channel"] == 1
    assert stats["logical_microbatches"] == 2
    assert stats["samples_total"] == 3
    assert stats["routing_valid_samples"] == 2
    assert stats["routing_invalid_samples"] == 1
    assert stats["routing_valid_fraction"] == pytest.approx(2 / 3)
    assert stats["tokens_real"] == 12
    assert stats["tokens_padded"] == 14
    assert stats["tokens_padding"] == 2


def test_begin_r3_microbatch_replay_counts_no_router_stage():
    routed = torch.ones(1, 4, 2, 2, dtype=torch.int32)
    engine = _fake_engine(routed)
    engine._r3_router_groups = {0: []}
    context = engine._prepare_r3_forward_context(_fake_mb_list(batch_size=1))

    replay = engine._begin_r3_microbatch_replay(
        context,
        vp_stage=0,
        mb_idx=0,
        mb_input=SimpleNamespace(orig_mb={}),
        forward_only=True,
        sequence_parallel=False,
    )

    assert replay is None
    assert context["stats"]["mode_no_router_microbatches"] == 1
    assert context["stats"]["no_router_real_tokens"] == 1
    assert context["stats"]["router_stage_microbatches"] == 0


def test_begin_r3_microbatch_replay_counts_invalid_record_fallback(monkeypatch):
    routed = torch.ones(1, 4, 2, 2, dtype=torch.int32)
    engine = _fake_engine(routed, routing_valid=torch.tensor([False]))
    ref = object()
    engine._r3_router_groups = {0: [ref]}
    context = engine._prepare_r3_forward_context(_fake_mb_list(batch_size=1))
    actions = []
    monkeypatch.setattr(
        "areal.engine.megatron_engine.set_router_replay_action",
        lambda refs, action: actions.append((refs, action)),
    )

    replay_refs, mode = engine._begin_r3_microbatch_replay(
        context,
        vp_stage=0,
        mb_idx=0,
        mb_input=SimpleNamespace(orig_mb={}),
        forward_only=False,
        sequence_parallel=False,
    )

    assert replay_refs == [ref]
    assert mode == "record"
    assert actions == [([ref], "RECORD")]
    stats = context["stats"]
    assert stats["router_stage_microbatches"] == 1
    assert stats["skipped_microbatches"] == 1
    assert stats["invalid_samples"] == 1
    assert stats["skip_invalid_sample"] == 1
    assert stats["mode_record_microbatches"] == 1
    assert stats["record_real_tokens"] == 1


def test_initialize_r3_router_replay_rejects_effective_moe_router_fusion():
    engine = object.__new__(MegatronEngine)
    engine._r3_enabled = True
    engine.model = []
    engine.tf_config = SimpleNamespace(moe_router_fusion=True)

    with pytest.raises(R3Error, match="moe_router_fusion=True"):
        engine._initialize_r3_router_replay()
