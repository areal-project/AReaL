# SPDX-License-Identifier: Apache-2.0

import types

import areal.engine.megatron_utils.mindspeed_gdn_patch as gdn_patch
from areal.engine.megatron_utils.mindspeed_gdn_patch import (
    ensure_mindspeed_gdn_conv1d,
    has_mindspeed_gdn_conv1d,
    has_mindspeed_gdn_model_classes,
)


def test_ensure_mindspeed_gdn_conv1d_binds_selected_backend(monkeypatch):
    original_conv = object()
    npu_conv = object()
    modules = {
        "megatron.core.ssm.gated_delta_net": types.SimpleNamespace(
            causal_conv1d=original_conv
        ),
        "mindspeed.core.ssm.gated_delta_net": types.SimpleNamespace(),
        "mindspeed.core.ssm.gdn_backend": types.SimpleNamespace(
            load_gdn_operators=lambda: types.SimpleNamespace(causal_conv1d=npu_conv)
        ),
    }
    monkeypatch.setattr(gdn_patch.importlib, "import_module", modules.__getitem__)

    assert ensure_mindspeed_gdn_conv1d()
    assert modules["megatron.core.ssm.gated_delta_net"].causal_conv1d is npu_conv
    assert modules["mindspeed.core.ssm.gated_delta_net"].causal_conv1d is npu_conv


def test_ensure_mindspeed_gdn_conv1d_is_idempotent(monkeypatch):
    npu_conv = object()
    modules = {
        "megatron.core.ssm.gated_delta_net": types.SimpleNamespace(),
        "mindspeed.core.ssm.gated_delta_net": types.SimpleNamespace(),
        "mindspeed.core.ssm.gdn_backend": types.SimpleNamespace(
            load_gdn_operators=lambda: types.SimpleNamespace(causal_conv1d=npu_conv)
        ),
    }
    monkeypatch.setattr(gdn_patch.importlib, "import_module", modules.__getitem__)

    assert ensure_mindspeed_gdn_conv1d()
    assert ensure_mindspeed_gdn_conv1d()
    assert modules["megatron.core.ssm.gated_delta_net"].causal_conv1d is npu_conv
    assert modules["mindspeed.core.ssm.gated_delta_net"].causal_conv1d is npu_conv


def test_ensure_mindspeed_gdn_conv1d_requires_selected_backend(monkeypatch):
    modules = {
        "megatron.core.ssm.gated_delta_net": types.SimpleNamespace(),
        "mindspeed.core.ssm.gated_delta_net": types.SimpleNamespace(),
    }

    def import_module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(gdn_patch.importlib, "import_module", import_module)

    assert not ensure_mindspeed_gdn_conv1d()


def test_has_mindspeed_gdn_conv1d_handles_missing_module_attribute():
    assert not has_mindspeed_gdn_conv1d(types.SimpleNamespace())


def test_has_mindspeed_gdn_model_classes_rejects_captured_mcore_class():
    mindspeed_class = type("GatedDeltaNet", (), {})
    mindspeed_class.__module__ = "mindspeed.core.ssm.gated_delta_net"
    mcore_class = type("GatedDeltaNet", (), {})
    mcore_class.__module__ = "megatron.core.ssm.gated_delta_net"

    assert has_mindspeed_gdn_model_classes(mindspeed_class, mindspeed_class)
    assert not has_mindspeed_gdn_model_classes(mindspeed_class, mcore_class)


def test_ensure_mindspeed_gdn_conv1d_keeps_binding_when_backend_lacks_conv1d(
    monkeypatch,
):
    """Test that a backend without causal_conv1d leaves existing bindings intact."""
    original = object()
    modules = {
        "megatron.core.ssm.gated_delta_net": types.SimpleNamespace(
            causal_conv1d=original
        ),
        "mindspeed.core.ssm.gated_delta_net": types.SimpleNamespace(
            causal_conv1d=original
        ),
        "mindspeed.core.ssm.gdn_backend": types.SimpleNamespace(
            load_gdn_operators=types.SimpleNamespace
        ),
    }
    monkeypatch.setattr(gdn_patch.importlib, "import_module", modules.__getitem__)

    assert not ensure_mindspeed_gdn_conv1d()
    assert modules["megatron.core.ssm.gated_delta_net"].causal_conv1d is original
    assert modules["mindspeed.core.ssm.gated_delta_net"].causal_conv1d is original


def test_cu_seqlens_host_list_uses_cached_pointer():
    import torch

    from areal.engine.megatron_utils.mindspeed_gdn_patch import (
        _CU_HOST,
        cu_seqlens_host_list,
        remember_cu_seqlens,
    )

    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    remember_cu_seqlens(cu, host=[0, 4, 8])
    assert _CU_HOST[cu.data_ptr()] == [0, 4, 8]

    def _boom(*_args, **_kwargs):
        raise AssertionError("device sync")

    cu.detach = _boom  # type: ignore[method-assign]
    assert cu_seqlens_host_list(cu) == [0, 4, 8]


def test_known_cu_seqlens_patches_mindspeed_adapters(monkeypatch):
    import torch

    conv = types.SimpleNamespace(
        _query_start_loc=lambda cu: [int(v) for v in cu.detach().cpu().tolist()]
    )
    delta = types.SimpleNamespace(
        _as_int_list=lambda cu: [int(v) for v in cu.detach().cpu().tolist()],
        _prepare_packed_metadata=lambda cu, g, chunk_size: (
            cu.to(dtype=torch.int64),
            [int(v) for v in cu.detach().cpu().tolist()],
            None,
            None,
        ),
    )
    modules = {
        "mindspeed.core.ssm.npu_causal_conv1d": conv,
        "mindspeed.core.ssm.npu_chunk_gated_delta_rule": delta,
    }
    monkeypatch.setattr(gdn_patch.importlib, "import_module", modules.__getitem__)

    from areal.engine.megatron_utils.mindspeed_gdn_patch import (
        _known_cu_seqlens,
        cu_seqlens_host_list,
        remember_cu_seqlens,
    )

    assert _known_cu_seqlens()
    assert conv._query_start_loc is cu_seqlens_host_list
    assert delta._as_int_list is cu_seqlens_host_list

    cu = torch.tensor([0, 8], dtype=torch.int32)
    remember_cu_seqlens(cu, host=[0, 8])
    g = torch.zeros(1, 8, 2)
    converted, host, *_ = delta._prepare_packed_metadata(cu, g, 64)
    assert host == [0, 8]
    assert converted.dtype == torch.int64
    assert cu_seqlens_host_list(converted) == [0, 8]


def test_pad_packed_and_mb_to_reregister_cu_seqlens_host():
    import torch

    from areal.api.cli_args import MicroBatchSpec
    from areal.engine.megatron_utils.mindspeed_gdn_patch import cu_seqlens_host_list
    from areal.utils.data import MicroBatchList, pad_packed_tensor_dict

    data = {
        "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        "max_seqlen": 3,
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.int32),
    }
    padded, _, _, _ = pad_packed_tensor_dict(data, pad_to_length=256)
    cu = padded["cu_seqlens"]
    assert cu_seqlens_host_list(cu) == [0, 3, 256]

    mb = MicroBatchList(
        data={"cu_seqlens": cu},
        mb_spec=MicroBatchSpec(),
        mbs=[{"cu_seqlens": cu}],
        group_lens=[3],
        padded_mbs=[{"cu_seqlens": cu}],
        padding_lengths=[253],
        padded_to_lengths=[256],
    )
    moved = mb.to(dtype=torch.int64)
    moved_cu = moved.padded_mbs[0]["cu_seqlens"]
    assert moved_cu.dtype == torch.int64
    assert cu_seqlens_host_list(moved_cu) == [0, 3, 256]
