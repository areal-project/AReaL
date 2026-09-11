import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_kda_attention_module(monkeypatch):
    class _CudaRngTracker:
        def fork(self):
            return self

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class _Logger:
        def info(self, *args, **kwargs):
            return None

    _stub_module(monkeypatch, "megatron")
    _stub_module(monkeypatch, "megatron.core")
    _stub_module(
        monkeypatch,
        "megatron.core.parallel_state",
        model_parallel_is_initialized=lambda: False,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.dist_checkpointing",
        ShardedTensor=type("ShardedTensor", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.dist_checkpointing.mapping",
        ReplicaId=type("ReplicaId", (), {}),
        ShardedTensorFactory=type("ShardedTensorFactory", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.packed_seq_params",
        PackedSeqParams=type("PackedSeqParams", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.tensor_parallel",
        get_cuda_rng_tracker=lambda: _CudaRngTracker(),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.tensor_parallel.mappings",
        gather_from_tensor_model_parallel_region=lambda x: x,
        scatter_to_sequence_parallel_region=lambda x: x,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer",
        TransformerConfig=type("TransformerConfig", (), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.identity_op",
        IdentityOp=type("IdentityOp", (nn.Module,), {}),
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.module",
        MegatronModule=_MegatronModule,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.spec_utils",
        ModuleSpec=type("ModuleSpec", (), {}),
        build_module=lambda *args, **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "megatron.core.transformer.utils",
        make_sharded_tensors_for_checkpoint=lambda *args, **kwargs: {},
        sharded_state_dict_default=lambda *args, **kwargs: {},
    )
    _stub_module(monkeypatch, "areal")
    _stub_module(monkeypatch, "areal.utils")
    _stub_module(
        monkeypatch,
        "areal.utils.logging",
        getLogger=lambda name: _Logger(),
    )

    path = Path(__file__).resolve().parents[1] / "areal/models/mcore/kda_attention.py"
    spec = importlib.util.spec_from_file_location("_test_kda_attention", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_parameter_local_cp_slices_each_qkv_section(monkeypatch):
    kda = _load_kda_attention_module(monkeypatch)
    param = torch.arange(12).reshape(12, 1)

    sliced = kda._get_parameter_local_cp(
        param,
        dim=0,
        cp_rank=1,
        cp_size=2,
        split_size_or_sections=[4, 4, 4],
    )

    torch.testing.assert_close(sliced.squeeze(1), torch.tensor([2, 3, 6, 7, 10, 11]))


def test_all_to_all_cp2hp_preserves_qkv_section_boundaries(monkeypatch):
    kda = _load_kda_attention_module(monkeypatch)
    monkeypatch.setattr(kda.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(kda, "_all_to_all_equal", lambda input_, cp_group: input_)

    qkv = torch.arange(2 * 1 * 12).reshape(2, 1, 12)
    split_out = kda._all_to_all_cp2hp(
        qkv,
        cp_group=object(),
        split_size_or_sections=[4, 4, 4],
    )
    expected = torch.cat(
        [
            kda._all_to_all_cp2hp(chunk, cp_group=object())
            for chunk in torch.split(qkv, [4, 4, 4], dim=-1)
        ],
        dim=-1,
    )
    unsplit_out = kda._all_to_all_cp2hp(qkv, cp_group=object())

    torch.testing.assert_close(split_out, expected)
    assert not torch.equal(split_out, unsplit_out)


def test_out_norm_sharded_state_dict_uses_replicated_wrapper(monkeypatch):
    kda = _load_kda_attention_module(monkeypatch)

    class _FakeShard:
        def __init__(self, data):
            self.data = data

    class _OutNormWithOwnShard(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2))

        def sharded_state_dict(self, *args, **kwargs):
            raise AssertionError("out_norm own sharded_state_dict should be bypassed")

    calls = []

    def fake_make_sharded(
        state_dict,
        prefix,
        tensor_parallel_layers_axis_map=None,
        sharded_offsets=(),
    ):
        calls.append(
            (
                "make",
                prefix,
                tuple(state_dict),
                dict(tensor_parallel_layers_axis_map or {}),
            )
        )
        return {
            f"{prefix}{name}": _FakeShard(tensor) for name, tensor in state_dict.items()
        }

    def fake_default(module, prefix="", sharded_offsets=(), metadata=None):
        calls.append(("default", prefix))
        return {
            f"{prefix}{name}": _FakeShard(tensor)
            for name, tensor in module.state_dict(prefix="", keep_vars=True).items()
        }

    monkeypatch.setattr(kda, "make_sharded_tensors_for_checkpoint", fake_make_sharded)
    monkeypatch.setattr(kda, "sharded_state_dict_default", fake_default)
    monkeypatch.setattr(kda, "_split_tensor_factory", lambda shard, *args: shard)

    attn = kda.KimiDeltaAttention.__new__(kda.KimiDeltaAttention)
    nn.Module.__init__(attn)
    attn.tp_size = 2
    attn.in_proj_dim = 20
    attn.qk_dim = 4
    attn.v_dim = 4
    attn.value_head_dim = 2
    attn.no_kda_lora = True
    attn.conv_dim_local_tp = 6
    attn.conv_bias = False
    attn.add_module("in_proj", nn.Linear(1, 10, bias=False))
    attn.add_module("conv1d", nn.Conv1d(6, 6, 1, groups=6, bias=False))
    attn.add_module("out_norm", _OutNormWithOwnShard())

    state_dict = attn.sharded_state_dict(prefix="decoder.layers.0.self_attention.")

    assert "decoder.layers.0.self_attention.out_norm.weight" in state_dict
    assert (
        "make",
        "decoder.layers.0.self_attention.out_norm.",
        ("weight",),
        {},
    ) in calls
    assert ("default", "decoder.layers.0.self_attention.out_norm.") not in calls
