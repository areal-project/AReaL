from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("awex.util.tensor_util")
pytest.importorskip("awex.meta.weight_meta")
pytest.importorskip("awex.sharding.param_sharding")
pytest.importorskip("awex.transfer.transfer_plan")

from awex.util.tensor_util import reconstruct_tensors_from_groups

from areal.v2.weight_update.awex.megatron_adapter import (
    AwexMegatronAdapter,
    _install_awex_qwen2_converter,
)


def test_qwen2_converter_uses_unfused_hf_attention_names():
    from awex.models.registry import ModelRegistry

    original = ModelRegistry.models.get("Qwen2ForCausalLM")
    try:
        _install_awex_qwen2_converter()
        assert "Qwen2ForCausalLM" in ModelRegistry.models
        model_config = ModelRegistry.get_model_config("Qwen2ForCausalLM")
        converter_cls = (
            model_config["mcore_converter"]
            if isinstance(model_config, dict)
            else model_config.mcore_converter
        )
        converter = object.__new__(converter_cls)

        assert not converter._fuse_qkv("self_attention.linear_qkv.weight")
        assert (
            converter._normalize_attn_name("self_attn.o_proj.weight")
            == "self_attn.o_proj.weight"
        )
    finally:
        if original is None:
            ModelRegistry.models.pop("Qwen2ForCausalLM", None)
        else:
            ModelRegistry.models["Qwen2ForCausalLM"] = original


def test_get_weight_metadata_restores_legacy_offloaded_weights(monkeypatch):
    calls = []

    class _LegacyAdapter:
        _released_tags = {"weights", "optimizer"}

        def resume_memory(self, tags):
            calls.append(("resume", tags))

        def release_memory(self, tags):
            calls.append(("release", tags))

    adapter = object.__new__(AwexMegatronAdapter)
    adapter._engine = SimpleNamespace(_awex_adapter=_LegacyAdapter())
    adapter._released_tags = set()
    adapter._awex_train_meta = None
    expected = [object()]
    monkeypatch.setattr(adapter, "_ensure_awex_converter", lambda _: (expected, None))

    assert adapter.get_weight_metadata({"model": "test"}) is expected
    assert calls == [("resume", ["weights"]), ("release", ["weights"])]


class _BaseOptimizer:
    def __init__(self):
        self.param = object()
        self.state = {self.param: {"step": torch.tensor(1.0)}}


class _OptimizerWrapper:
    def __init__(self, base):
        self.optimizer = base


class _AssertingChainedOptimizer:
    def __init__(self, children):
        self.chained_optimizers = children

    @property
    def optimizer(self):
        raise AssertionError(
            "ChainedOptimizer has more than one optimizer when accessing self.optimizer"
        )


class _Engine:
    def __init__(self, optimizer):
        self.optimizer = optimizer


def test_optimizer_state_offload_flattens_chained_optimizer_without_optimizer_attr():
    child_a = _OptimizerWrapper(_BaseOptimizer())
    child_b = _OptimizerWrapper(_BaseOptimizer())
    chained = _AssertingChainedOptimizer([child_a, child_b])
    adapter = AwexMegatronAdapter(_Engine(chained))

    assert adapter._get_inner_optimizers() == [child_a, child_b]

    adapter._offload_optimizer_states()
    adapter._reload_optimizer_states()
    assert adapter._offloaded_optimizer_states == {}


def test_colocate_full_ipc_grouping_owns_storage_and_reconstructs_order():
    contiguous = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    adapter = object.__new__(AwexMegatronAdapter)
    adapter._live_module_storage_ptrs = lambda: {
        contiguous.untyped_storage().data_ptr()
    }

    groups, metadata = adapter._full_tensors_for_ipc([contiguous, noncontiguous])

    live_storage = contiguous.untyped_storage().data_ptr()
    assert all(g.untyped_storage().data_ptr() != live_storage for g in groups)
    assert all(g.is_contiguous() for g in groups)

    reconstructed = reconstruct_tensors_from_groups(groups, metadata)
    assert torch.equal(reconstructed[0], contiguous)
    assert torch.equal(reconstructed[1], noncontiguous)


def test_delta_sync_full_reason_promotes_peer_full_sync(monkeypatch):
    adapter = object.__new__(AwexMegatronAdapter)

    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.torch.cuda.is_available",
        lambda: False,
    )

    def fake_all_reduce(tensor, op):
        del op
        tensor.fill_(1)

    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.all_reduce",
        fake_all_reduce,
    )

    assert adapter._delta_sync_full_reason(None, version=4) == "global_full_sync"


def test_delta_sync_full_reason_keeps_global_delta(monkeypatch):
    adapter = object.__new__(AwexMegatronAdapter)

    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.torch.cuda.is_available",
        lambda: False,
    )

    def fake_all_reduce(tensor, op):
        del op
        assert int(tensor.item()) == 0

    monkeypatch.setattr(
        "areal.v2.weight_update.awex.megatron_adapter.dist.all_reduce",
        fake_all_reduce,
    )

    assert adapter._delta_sync_full_reason(None, version=4) is None
