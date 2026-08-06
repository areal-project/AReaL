from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_remap_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "areal"
        / "v2"
        / "weight_update"
        / "awex"
        / "delta_index_remap.py"
    )
    spec = importlib.util.spec_from_file_location("delta_index_remap", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REMAP = _load_remap_module()
Qwen3MoeIndexRemapConfig = _REMAP.Qwen3MoeIndexRemapConfig
MCoreShardDirtyBitset = _REMAP.MCoreShardDirtyBitset
remap_qwen3_moe_mcore_indices_to_hf = _REMAP.remap_qwen3_moe_mcore_indices_to_hf
remap_qwen3_moe_mcore_bitset_to_hf = _REMAP.remap_qwen3_moe_mcore_bitset_to_hf
remap_qwen3_moe_mcore_shard_bitset_to_hf = (
    _REMAP.remap_qwen3_moe_mcore_shard_bitset_to_hf
)
remap_qwen3_moe_mcore_shard_bitsets_to_hf_masks = (
    _REMAP.remap_qwen3_moe_mcore_shard_bitsets_to_hf_masks
)


def _cfg() -> Qwen3MoeIndexRemapConfig:
    return Qwen3MoeIndexRemapConfig(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        head_dim=2,
    )


def _as_dict(remapped):
    return {item.name: item.indices for item in remapped}


def _pack_indices(indices: torch.Tensor, numel: int) -> torch.Tensor:
    mask = torch.zeros(numel, dtype=torch.bool)
    mask[indices.to(torch.long)] = True
    padded = ((numel + 7) // 8) * 8
    if padded != numel:
        mask = torch.cat((mask, torch.zeros(padded - numel, dtype=torch.bool)))
    bits = mask.view(-1, 8).to(torch.uint8)
    return sum(bits[:, bit] << bit for bit in range(8))


def _assert_same_remap_from_bitset(
    name: str,
    indices: torch.Tensor,
    shape: tuple[int, ...],
    *,
    packed: torch.Tensor | None = None,
    chunk_bytes: int = 1,
) -> None:
    expected = _as_dict(
        remap_qwen3_moe_mcore_indices_to_hf(name, indices, shape, _cfg())
    )
    actual = _as_dict(
        remap_qwen3_moe_mcore_bitset_to_hf(
            name,
            _pack_indices(indices, int(torch.tensor(shape).prod()))
            if packed is None
            else packed,
            shape,
            _cfg(),
            chunk_bytes=chunk_bytes,
        )
    )

    assert set(actual) == set(expected)
    for key, expected_indices in expected.items():
        assert torch.equal(actual[key], expected_indices)


def test_remap_identity_parameter_keeps_flat_indices() -> None:
    indices = torch.tensor([0, 3, 7], dtype=torch.int32)

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        indices,
        (8, 8),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert set(by_name) == {"model.layers.0.self_attn.o_proj.weight"}
    assert torch.equal(by_name["model.layers.0.self_attn.o_proj.weight"], indices)


def test_remap_dense_mlp_linear_fc1_splits_gate_and_up_rows() -> None:
    indices = torch.tensor([1, 8, 17, 31], dtype=torch.int32)

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.3.mlp.linear_fc1.weight",
        indices,
        (4, 8),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert torch.equal(
        by_name["model.layers.3.mlp.gate_proj.weight"],
        torch.tensor([1, 8], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.3.mlp.up_proj.weight"],
        torch.tensor([1, 15], dtype=torch.int32),
    )


def test_remap_expert_linear_fc1_splits_gate_and_up_rows() -> None:
    indices = torch.tensor([0, 16, 31], dtype=torch.int32)

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.5.mlp.experts.linear_fc1.weight12",
        indices,
        (4, 8),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert torch.equal(
        by_name["model.layers.5.mlp.experts.12.gate_proj.weight"],
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.5.mlp.experts.12.up_proj.weight"],
        torch.tensor([0, 15], dtype=torch.int32),
    )


def test_remap_qkv_weight_matches_qwen3_moe_row_permutation() -> None:
    # cfg: G=2, value_num_per_group=2, D=2, H=8
    # mcore rows per group: q0,q1,k,v where each slot has D rows.
    indices = torch.tensor(
        [
            0 * 8 + 3,  # group0 q slot0 d0 -> q row 0, col 3
            3 * 8 + 4,  # group0 q slot1 d1 -> q row 3, col 4
            4 * 8 + 5,  # group0 k d0 -> k row 0, col 5
            7 * 8 + 6,  # group0 v d1 -> v row 1, col 6
            8 * 8 + 7,  # group1 q slot0 d0 -> q row 4, col 7
            12 * 8 + 1,  # group1 k d0 -> k row 2, col 1
            15 * 8 + 2,  # group1 v d1 -> v row 3, col 2
        ],
        dtype=torch.int32,
    )

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.1.self_attention.linear_qkv.weight",
        indices,
        (16, 8),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert torch.equal(
        by_name["model.layers.1.self_attn.q_proj.weight"],
        torch.tensor([3, 28, 39], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.k_proj.weight"],
        torch.tensor([5, 17], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.v_proj.weight"],
        torch.tensor([14, 26], dtype=torch.int32),
    )


def test_remap_qkv_weight_accepts_tp_local_groups() -> None:
    # Local TP shards use the same GQA group layout as AWEX's Qwen3-MoE
    # converter, but only contain a subset of the KV groups.
    indices = torch.tensor(
        [
            0 * 8 + 3,  # local group0 q slot0 d0 -> local q row 0, col 3
            3 * 8 + 4,  # local group0 q slot1 d1 -> local q row 3, col 4
            4 * 8 + 5,  # local group0 k d0 -> local k row 0, col 5
            7 * 8 + 6,  # local group0 v d1 -> local v row 1, col 6
        ],
        dtype=torch.int32,
    )

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.1.self_attention.linear_qkv.weight",
        indices,
        (8, 8),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert torch.equal(
        by_name["model.layers.1.self_attn.q_proj.weight"],
        torch.tensor([3, 28], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.k_proj.weight"],
        torch.tensor([5], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.v_proj.weight"],
        torch.tensor([14], dtype=torch.int32),
    )


def test_remap_qkv_bias_matches_qwen3_moe_split() -> None:
    indices = torch.tensor([0, 3, 4, 7, 8, 12, 15], dtype=torch.int32)

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.1.self_attention.linear_qkv.bias",
        indices,
        (16,),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert torch.equal(
        by_name["model.layers.1.self_attn.q_proj.bias"],
        torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.k_proj.bias"],
        torch.tensor([0, 2], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.v_proj.bias"],
        torch.tensor([1, 3], dtype=torch.int32),
    )


def test_remap_qkv_bias_accepts_tp_local_groups() -> None:
    indices = torch.tensor([0, 3, 4, 7], dtype=torch.int32)

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.1.self_attention.linear_qkv.bias",
        indices,
        (8,),
        _cfg(),
    )

    by_name = _as_dict(result)
    assert torch.equal(
        by_name["model.layers.1.self_attn.q_proj.bias"],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.k_proj.bias"],
        torch.tensor([0], dtype=torch.int32),
    )
    assert torch.equal(
        by_name["model.layers.1.self_attn.v_proj.bias"],
        torch.tensor([1], dtype=torch.int32),
    )


def test_remap_empty_dirty_indices_returns_empty_result() -> None:
    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        torch.tensor([], dtype=torch.int32),
        (8, 8),
        _cfg(),
    )

    assert result == []


def test_remap_unsupported_parameter_name_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported Qwen3-MoE"):
        remap_qwen3_moe_mcore_indices_to_hf(
            "module.module.decoder.layers.0.unknown.weight",
            torch.tensor([0], dtype=torch.int32),
            (8,),
            _cfg(),
        )


def test_remap_qkv_weight_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="qkv rows mismatch"):
        remap_qwen3_moe_mcore_indices_to_hf(
            "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
            torch.tensor([0], dtype=torch.int32),
            (15, 8),
            _cfg(),
        )


def test_remap_expert_local_id_uses_ep_global_offset() -> None:
    cfg = Qwen3MoeIndexRemapConfig(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        head_dim=2,
        num_moe_experts=8,
        expert_model_parallel_size=4,
        expert_model_parallel_rank=2,
    )

    result = remap_qwen3_moe_mcore_indices_to_hf(
        "module.module.decoder.layers.5.mlp.experts.linear_fc1.weight1",
        torch.tensor([0, 16], dtype=torch.int32),
        (4, 8),
        cfg,
    )

    by_name = _as_dict(result)
    assert set(by_name) == {
        "model.layers.5.mlp.experts.5.gate_proj.weight",
        "model.layers.5.mlp.experts.5.up_proj.weight",
    }


def test_bitset_remap_identity_ignores_padding_bits() -> None:
    indices = torch.tensor([0, 3, 9], dtype=torch.int32)
    packed = _pack_indices(indices, 10)
    packed[-1] |= 0b1111_1100

    _assert_same_remap_from_bitset(
        "module.module.decoder.layers.0.self_attention.linear_proj.weight",
        indices,
        (2, 5),
        packed=packed,
        chunk_bytes=1,
    )


def test_bitset_remap_row_split_matches_indices_path() -> None:
    _assert_same_remap_from_bitset(
        "module.module.decoder.layers.3.mlp.linear_fc1.weight",
        torch.tensor([1, 8, 17, 31], dtype=torch.int32),
        (4, 8),
        chunk_bytes=1,
    )


def test_bitset_remap_qkv_weight_matches_indices_path() -> None:
    indices = torch.tensor(
        [
            0 * 8 + 3,
            3 * 8 + 4,
            4 * 8 + 5,
            7 * 8 + 6,
            8 * 8 + 7,
            12 * 8 + 1,
            15 * 8 + 2,
        ],
        dtype=torch.int32,
    )

    _assert_same_remap_from_bitset(
        "module.module.decoder.layers.1.self_attention.linear_qkv.weight",
        indices,
        (16, 8),
        chunk_bytes=1,
    )


def test_bitset_remap_numel_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="numel mismatch"):
        remap_qwen3_moe_mcore_bitset_to_hf(
            "module.module.decoder.layers.0.self_attention.linear_proj.weight",
            torch.zeros(2, dtype=torch.uint8),
            (2, 5),
            _cfg(),
            numel=9,
        )


def test_shard_bitset_remap_offsets_local_indices() -> None:
    name = "module.module.decoder.layers.0.self_attention.linear_proj.weight"
    full_indices = torch.tensor([9, 11, 14], dtype=torch.int32)
    local_indices = full_indices - 8
    expected = _as_dict(
        remap_qwen3_moe_mcore_indices_to_hf(
            name,
            full_indices,
            (4, 4),
            _cfg(),
        )
    )
    actual = _as_dict(
        remap_qwen3_moe_mcore_shard_bitset_to_hf(
            name,
            _pack_indices(local_indices, 8),
            (4, 4),
            _cfg(),
            shard_start=8,
            shard_numel=8,
            chunk_bytes=1,
        )
    )

    assert set(actual) == set(expected)
    for key, expected_indices in expected.items():
        assert torch.equal(actual[key], expected_indices)


def test_shard_bitset_remap_qkv_weight_matches_full_indices_path() -> None:
    name = "module.module.decoder.layers.1.self_attention.linear_qkv.weight"
    full_indices = torch.tensor(
        [
            4 * 8 + 5,
            7 * 8 + 6,
            8 * 8 + 7,
            12 * 8 + 1,
        ],
        dtype=torch.int32,
    )
    shard_start = 4 * 8
    shard_numel = 9 * 8
    local_indices = full_indices - shard_start
    expected = _as_dict(
        remap_qwen3_moe_mcore_indices_to_hf(
            name,
            full_indices,
            (16, 8),
            _cfg(),
        )
    )
    actual = _as_dict(
        remap_qwen3_moe_mcore_shard_bitset_to_hf(
            name,
            _pack_indices(local_indices, shard_numel),
            (16, 8),
            _cfg(),
            shard_start=shard_start,
            shard_numel=shard_numel,
            chunk_bytes=2,
        )
    )

    assert set(actual) == set(expected)
    for key, expected_indices in expected.items():
        assert torch.equal(actual[key], expected_indices)


def test_shard_bitset_remap_range_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        remap_qwen3_moe_mcore_shard_bitset_to_hf(
            "module.module.decoder.layers.0.self_attention.linear_proj.weight",
            torch.zeros(2, dtype=torch.uint8),
            (2, 5),
            _cfg(),
            shard_start=8,
            shard_numel=3,
        )


def test_shard_bitsets_to_hf_masks_merge_and_fill_empty() -> None:
    name = "module.module.decoder.layers.3.mlp.linear_fc1.weight"
    records = [
        MCoreShardDirtyBitset(
            name=name,
            packed_bitset=_pack_indices(torch.tensor([1, 3], dtype=torch.int32), 8),
            shape=(4, 8),
            shard_start=0,
            shard_numel=8,
        ),
        MCoreShardDirtyBitset(
            name=name,
            packed_bitset=_pack_indices(torch.tensor([1, 7], dtype=torch.int32), 8),
            shape=(4, 8),
            shard_start=16,
            shard_numel=8,
        ),
    ]

    masks = remap_qwen3_moe_mcore_shard_bitsets_to_hf_masks(
        records,
        _cfg(),
        hf_names=[
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.3.mlp.up_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
        ],
    )

    assert torch.equal(
        masks["model.layers.3.mlp.gate_proj.weight"],
        torch.tensor([1, 3], dtype=torch.int32),
    )
    assert torch.equal(
        masks["model.layers.3.mlp.up_proj.weight"],
        torch.tensor([1, 7], dtype=torch.int32),
    )
    assert masks["model.layers.3.mlp.down_proj.weight"].dtype == torch.int32
    assert masks["model.layers.3.mlp.down_proj.weight"].numel() == 0


def test_shard_bitsets_to_hf_masks_fill_only_covered_names() -> None:
    name = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight1"
    cfg = Qwen3MoeIndexRemapConfig(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        head_dim=2,
        num_moe_experts=8,
        expert_model_parallel_size=4,
        expert_model_parallel_rank=2,
    )
    records = [
        MCoreShardDirtyBitset(
            name=name,
            packed_bitset=_pack_indices(torch.tensor([], dtype=torch.int32), 32),
            shape=(4, 8),
            shard_start=0,
            shard_numel=32,
        )
    ]

    masks = remap_qwen3_moe_mcore_shard_bitsets_to_hf_masks(
        records,
        cfg,
        hf_names=[
            "model.layers.3.mlp.experts.5.gate_proj.weight",
            "model.layers.3.mlp.experts.5.up_proj.weight",
            "model.layers.3.self_attn.q_proj.weight",
        ],
        fill_all_hf_names=False,
    )

    assert set(masks) == {
        "model.layers.3.mlp.experts.5.gate_proj.weight",
        "model.layers.3.mlp.experts.5.up_proj.weight",
    }
    assert masks["model.layers.3.mlp.experts.5.gate_proj.weight"].numel() == 0
    assert masks["model.layers.3.mlp.experts.5.up_proj.weight"].numel() == 0
