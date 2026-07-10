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
    engine._r3_pending_valid = torch.ones(
        routed_experts.shape[0],
        dtype=torch.bool,
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


def test_initialize_r3_router_replay_rejects_effective_moe_router_fusion():
    engine = object.__new__(MegatronEngine)
    engine._r3_enabled = True
    engine.model = []
    engine.tf_config = SimpleNamespace(moe_router_fusion=True)

    with pytest.raises(R3Error, match="moe_router_fusion=True"):
        engine._initialize_r3_router_replay()
