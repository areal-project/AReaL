# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.layout import prepare_native_replay_slabs
from areal.engine.r3.packed import PackedRouting, compact_routing_dtype
from areal.engine.r3.preprocess import preprocess_routed_experts_batch
from areal.engine.r3.transport import pop_r3_tensors, without_r3_tensors
from areal.utils.data import concat_padded_tensors, split_and_unpad_tensor


@pytest.mark.parametrize(
    "maximum,dtype", [(511, torch.int16), (32767, torch.int16), (32768, torch.int32)]
)
def test_compact_dtype_preserves_expert_ids(maximum, dtype):
    original = torch.tensor([0, maximum], dtype=torch.int32)
    result = compact_routing_dtype(original)
    assert result.dtype == dtype
    torch.testing.assert_close(result.int(), original, rtol=0, atol=0)


def test_compact_dtype_rejects_negative_expert_id():
    with pytest.raises(R3Error, match="nonnegative"):
        compact_routing_dtype(torch.tensor([-1, 0], dtype=torch.int32))


def _trajectory(length):
    raw = np.arange(1, length * 6 + 1, dtype=np.int32).reshape(length, 3, 2)
    result = preprocess_routed_experts_batch(
        [raw], seq_lens=[length], num_moe_layers=3, topk=2, packed=True
    )
    result["attention_mask"] = torch.ones(1, length, dtype=torch.bool)
    return result


def test_concat_and_split_preserve_packed_routing_without_sequence_padding():
    trajectories = [_trajectory(3), _trajectory(7)]
    batch = concat_padded_tensors(trajectories)
    assert batch["routed_experts"].shape == (10, 3, 2)
    assert batch["routed_experts"].dtype == torch.int16
    assert batch["r3_cu_seqlens"].tolist() == [0, 3, 10]
    restored = split_and_unpad_tensor(batch, 2)
    for actual, original in zip(restored, trajectories, strict=True):
        for key in original:
            torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
    routes, valid = pop_r3_tensors(batch)
    assert isinstance(routes, PackedRouting)
    assert valid.tolist() == [True, True]
    assert set(without_r3_tensors(trajectories[0])) == {"attention_mask"}


def test_row_selection_composes_without_copying_payload():
    routes = PackedRouting.from_tensors(
        torch.arange(36, dtype=torch.int16).reshape(6, 3, 2),
        torch.tensor([0, 1, 3, 6]),
    )
    selected = routes.select_rows([2, 0, 2]).select_rows([2, 1])
    assert selected.values.data_ptr() == routes.values.data_ptr()
    assert selected.row_indices.tolist() == [2, 0]
    torch.testing.assert_close(selected.row(0), routes.row(2), rtol=0, atol=0)


@pytest.mark.parametrize("offsets", [[1, 2], [0, 3, 2], [0, 1]])
def test_packed_routes_reject_invalid_offsets(offsets):
    with pytest.raises(R3Error, match="offsets"):
        PackedRouting.from_tensors(
            torch.ones(2, 1, 1, dtype=torch.int16), torch.tensor(offsets)
        )


def test_packed_routes_reject_fractional_offsets_before_cast():
    with pytest.raises(R3Error, match="integer dtype"):
        PackedRouting.from_tensors(
            torch.ones(3, 1, 1, dtype=torch.int16), torch.tensor([0.0, 1.9, 3.0])
        )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_packed_local_gather_matches_dense_layout_all_parallel_ranks(
    monkeypatch, cp_size, tp_size, sequence_parallel
):
    """Use the old dense implementation as an independent layout oracle."""
    from megatron.core import parallel_state as mpu

    dense = torch.arange(2 * 7 * 3 * 2, dtype=torch.int32).reshape(2, 7, 3, 2)
    values = torch.cat([dense[0, :5], dense[1, :7]]).short()
    routes = PackedRouting.from_tensors(values, torch.tensor([0, 5, 12]))
    old = torch.tensor([0, 5, 12], dtype=torch.int32)
    # Includes per-sequence padding and a trailing batch padding segment.
    padded = torch.tensor([0, 32, 64, 96], dtype=torch.int32)
    mb = SimpleNamespace(
        orig_mb={"cu_seqlens": old},
        padded_mb={"cu_seqlens": padded},
        old_cu_seqlens=old,
    )
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)
    for cp_rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
        for tp_rank in range(tp_size):
            monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: tp_rank)
            group = SimpleNamespace(size=lambda: tp_size, rank=lambda: tp_rank)
            monkeypatch.setattr(
                "megatron.core.tensor_parallel.mappings.get_tensor_model_parallel_group_if_none",
                lambda _: group,
            )
            expected = prepare_native_replay_slabs(
                dense,
                torch.tensor([True, True]),
                mb,
                [2, 0],
                sequence_parallel=sequence_parallel,
            )
            actual = prepare_native_replay_slabs(
                routes,
                torch.tensor([True, True]),
                mb,
                [2, 0],
                sequence_parallel=sequence_parallel,
            )
            assert not actual.skip_replay
            for left, right in zip(actual.slabs, expected.slabs, strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_layer_selection_owns_only_selected_layer_storage():
    routes = PackedRouting.from_tensors(
        torch.ones(10, 8, 2, dtype=torch.int16), torch.tensor([0, 10])
    )
    local = routes.select_layers([6, 1])
    assert local.layer_indices == (6, 1)
    assert local.values.untyped_storage().nbytes() == 10 * 2 * 2 * 2
    assert local.values.data_ptr() != routes.values.data_ptr()


@pytest.mark.parametrize("valid_dtype", [torch.bool, torch.int32, torch.float32])
def test_actor_and_engine_permutations_share_packed_payload(valid_dtype):
    """Compose the actual PPO and engine permutations, including duplicate rows."""
    from areal.engine.megatron_engine import MegatronEngine
    from areal.trainer.ppo.actor import PPOActor

    routes = PackedRouting.from_tensors(
        torch.arange(30, dtype=torch.int16).reshape(5, 3, 2), torch.tensor([0, 2, 5])
    )
    actor = object.__new__(PPOActor)
    actor.engine = SimpleNamespace(_r3_enabled=True)
    selected, valid = actor._split_r3_minibatches(
        routes,
        torch.tensor([True, False], dtype=valid_dtype),
        [{"_r3_row_indices": torch.tensor([[1], [0], [1]])}],
    )[0]
    engine = object.__new__(MegatronEngine)
    engine._r3_moe_config = SimpleNamespace(topk=2, num_moe_layers=3)
    engine._r3_num_moe_experts = 512
    mb_list = SimpleNamespace(
        forward_indices=[2, 0, 1],
        transport_dummy_count=0,
        mbs=[
            {"cu_seqlens": torch.tensor([0, 3, 6])},
            {"cu_seqlens": torch.tensor([0, 2])},
        ],
        group_lens=[6, 2],
        padding_lengths=[0, 0],
        padded_to_lengths=[6, 2],
    )
    context = engine._prepare_packed_r3_forward_context(selected, valid, mb_list)
    assert [mb.row_indices.tolist() for mb in context["routed_mbs"]] == [[1, 1], [0]]
    assert [mb.tolist() for mb in context["valid_mbs"]] == [[False, False], [True]]
    assert all(
        mb.values.data_ptr() == routes.values.data_ptr() for mb in context["routed_mbs"]
    )
    assert context["stats"]["cpu_routing_storage_bytes"] == routes.values.nbytes


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("remote", [False, True])
def test_packed_eval_dummy_preserves_group_and_layer_schema(active, remote):
    from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
    from areal.utils.data import make_dummy_eval_item

    template = concat_padded_tensors([_trajectory(3), _trajectory(7)])
    template["r3_layer_indices"] = torch.tensor([4, 1, 2])
    if remote:
        template = {
            key: RTensor(TensorShardInfo(key, "unused"), value.to("meta"))
            for key, value in template.items()
        }
    dummy = make_dummy_eval_item(template, active_attention=active)
    # Layer references can remain remote until the normal RPC localization.
    assert dummy["r3_layer_indices"] is template["r3_layer_indices"]
    dummy["r3_layer_indices"] = torch.tensor([4, 1, 2])
    routes, valid = pop_r3_tensors(dummy)
    assert routes.layer_indices == (4, 1, 2)
    assert routes.values.shape == (2 if active else 0, 3, 2)
    assert routes.cu_seqlens.tolist() == ([0, 1, 2] if active else [0, 0, 0])
    assert valid.tolist() == [False, False]
    assert dummy["attention_mask"].shape == (2, 1)
    assert dummy["attention_mask"].sum() == (2 if active else 0)
