from types import SimpleNamespace

import torch

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core import stage_batch_for_engine
from areal.infra.rpc.guard.engine_blueprint import (
    _should_stage_rpc_payload_on_cpu,
)
from areal.models.tree_attn import tree as tree_module
from areal.utils.data import MicroBatchItem, MicroBatchList


def _make_item(*, alias_orig_and_padded: bool = False) -> MicroBatchItem:
    orig_mb = {
        "loss_mask": torch.tensor([True, False]),
        "metadata": {"name": "sample"},
    }
    padded_mb = (
        orig_mb
        if alias_orig_and_padded
        else {"input_ids": torch.tensor([1, 2], dtype=torch.long)}
    )
    return MicroBatchItem(
        orig_mb=orig_mb,
        padded_mb=padded_mb,
        padding_length=0,
        old_cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        padded_to_length=2,
    )


def test_microbatch_item_to_moves_copy_without_mutating_source():
    """Device staging must not write accelerator tensors into the CPU source."""
    source = _make_item()

    staged = source.to("meta")

    assert staged.orig_mb["loss_mask"].device.type == "meta"
    assert staged.padded_mb["input_ids"].device.type == "meta"
    assert staged.old_cu_seqlens.device.type == "meta"
    assert source.orig_mb["loss_mask"].device.type == "cpu"
    assert source.padded_mb["input_ids"].device.type == "cpu"
    assert source.old_cu_seqlens.device.type == "cpu"


def test_microbatch_item_to_preserves_tree_input_alias():
    """Tree orig/padded payloads must be transferred once and remain aliased."""
    source = _make_item(alias_orig_and_padded=True)

    staged = source.to("meta")

    assert staged.orig_mb is staged.padded_mb
    assert staged.orig_mb is not source.orig_mb
    assert source.orig_mb is source.padded_mb


def test_virtual_pipeline_iterators_independently_stream_cpu_sources():
    """Each VP chunk must iterate the CPU list without retaining staged items."""
    source_item = _make_item()
    mb_list = MicroBatchList(
        data={"sentinel": torch.tensor([7])},
        mb_spec=MicroBatchSpec(),
        mbs=[source_item.orig_mb],
        group_lens=[2],
        padded_mbs=[source_item.padded_mb],
        padding_lengths=[0],
        padded_to_lengths=[2],
        old_cu_seqlens_list=[source_item.old_cu_seqlens],
    )

    first_vp_item = next(iter(mb_list)).to("meta")
    second_vp_item = next(iter(mb_list)).to("meta")

    assert first_vp_item.padded_mb["input_ids"].device.type == "meta"
    assert second_vp_item.padded_mb["input_ids"].device.type == "meta"
    assert first_vp_item.padded_mb is not second_vp_item.padded_mb
    assert mb_list.padded_mbs[0]["input_ids"].device.type == "cpu"
    assert mb_list.data["sentinel"].device.type == "cpu"


def test_stage_batch_for_streaming_engine_replaces_payload_in_place():
    """RPC aliases should observe replacement of the transient batch payload."""
    data = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "nested": (torch.tensor([3], dtype=torch.long),),
    }
    rpc_alias = data
    original_nested = data["nested"]
    engine = SimpleNamespace(stream_microbatches_from_cpu=True)

    result = stage_batch_for_engine(data, engine)

    assert result is data
    assert rpc_alias is data
    assert data["input_ids"].device.type == "cpu"
    assert data["nested"] is not original_nested
    assert isinstance(data["nested"], tuple)


def test_stage_batch_for_non_streaming_engine_keeps_payload_unchanged():
    """FSDP/Archon placement remains controlled by their existing paths."""
    data = {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}
    engine = SimpleNamespace(stream_microbatches_from_cpu=False)

    result = stage_batch_for_engine(data, engine)

    assert result is data


def test_rpc_payload_cpu_staging_is_limited_to_declared_methods():
    """Only methods declared by a streaming engine should use CPU broadcast."""
    engine = SimpleNamespace(cpu_staged_rpc_methods=frozenset({"ppo_update"}))

    assert _should_stage_rpc_payload_on_cpu(engine, "ppo_update")
    assert not _should_stage_rpc_payload_on_cpu(engine, "compute_advantages")
    assert not _should_stage_rpc_payload_on_cpu(SimpleNamespace(), "ppo_update")


def test_tree_count_collective_device_matches_backend(monkeypatch):
    group = object()
    monkeypatch.setattr(tree_module.dist, "get_backend", lambda _: "gloo")

    assert tree_module._tree_count_collective_device(group) == torch.device("cpu")

    monkeypatch.setattr(tree_module.dist, "get_backend", lambda _: "nccl")
    monkeypatch.setattr(tree_module.current_platform, "device_type", "cuda")
    monkeypatch.setattr(tree_module.current_platform, "current_device", lambda: 3)

    assert tree_module._tree_count_collective_device(group) == torch.device("cuda:3")
