# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch

import areal.engine.megatron_utils.mindspeed_gdn_patch as gdn_patch
from areal.engine.megatron_utils.mindspeed_gdn_patch import (
    ensure_mindspeed_gdn_conv1d,
    ensure_mindspeed_gdn_native_kkt,
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


@pytest.fixture
def native_kkt_modules(monkeypatch):
    """Exercise layout and metadata plumbing without loading NPU libraries."""
    calls = {}

    def original(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_out,
        chunk_size,
    ):
        raise AssertionError("Original forward was called")

    def cumsum(g, **kwargs):
        calls["cumsum"] = kwargs
        return g + 1

    def kkt(k, g, beta, **kwargs):
        calls["kkt"] = (k, g, beta, kwargs)
        b, h, t, _ = k.shape
        return torch.arange(
            b * h * t * kwargs["chunk_size"], dtype=torch.float32
        ).reshape(b, h, t, -1)

    def solve(a, **kwargs):
        calls["solve"] = (a, kwargs)
        return a + 1

    def wu(k, v, beta, a, chunk_size, **kwargs):
        calls["wu"] = (k, v, beta, a, kwargs)
        return k, v

    def state(k, w, u, **kwargs):
        calls["state"] = kwargs
        return k, u, object()

    def output(q, k, v, h, scale, **kwargs):
        calls["output"] = kwargs
        return v

    owner = types.SimpleNamespace(
        _chunk_gated_delta_rule_fwd=original,
        _chunk_gated_delta_rule_bwd=object(),
        ChunkGatedDeltaRuleFunction=object(),
        chunk_scaled_dot_kkt_fwd=object(),
        chunk_local_cumsum=cumsum,
        npu_solve_tri=solve,
        npu_recompute_w_u_fwd=wu,
        npu_chunk_gated_delta_rule_fwd_h=state,
        npu_chunk_fwd_o=output,
    )
    native = types.SimpleNamespace(npu_chunk_scaled_dot_kkt=kkt)
    modules = {
        "mindspeed.core.ssm.npu_chunk_gated_delta_rule": owner,
        "fla_npu.ops.ascendc": native,
        "mindspeed.core.ssm.gdn_backend": types.SimpleNamespace(
            load_gdn_operators=lambda: types.SimpleNamespace(name="fla_npu")
        ),
    }

    def import_module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(gdn_patch.importlib, "import_module", import_module)
    inputs = dict(
        q=torch.ones(1, 8, 2, 128, dtype=torch.bfloat16),
        k=torch.ones(1, 8, 2, 128, dtype=torch.bfloat16),
        v=torch.ones(1, 8, 2, 128, dtype=torch.bfloat16),
        g=torch.arange(16, dtype=torch.float32).reshape(1, 8, 2),
        beta=torch.ones(1, 8, 2, dtype=torch.bfloat16),
        scale=128**-0.5,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=torch.tensor([0, 3, 8]),
        cu_seqlens_list=[0, 3, 8],
        chunk_indices=[0, 0, 1, 0],
        chunk_indices_out={"64": torch.tensor([[0, 0], [1, 0]])},
        chunk_size=64,
    )
    return owner, modules, calls, inputs


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("chunk_size", [64, 128])
def test_native_kkt_preserves_layouts_and_metadata(
    native_kkt_modules, monkeypatch, packed, dtype, chunk_size
):
    """Saved A stays head-first and host metadata is reused without a copy."""
    owner, _, calls, inputs = native_kkt_modules
    for name in ("q", "k", "v", "beta"):
        inputs[name] = inputs[name].to(dtype)
    inputs["chunk_size"] = chunk_size
    if not packed:
        for name in (
            "cu_seqlens",
            "cu_seqlens_list",
            "chunk_indices",
            "chunk_indices_out",
        ):
            inputs[name] = None
    backward = owner._chunk_gated_delta_rule_bwd
    autograd = owner.ChunkGatedDeltaRuleFunction
    triton = owner.chunk_scaled_dot_kkt_fwd
    assert ensure_mindspeed_gdn_native_kkt()

    def forbid_host_copy(*args, **kwargs):
        raise AssertionError("Unexpected device-to-host conversion")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid_host_copy)
    monkeypatch.setattr(torch.Tensor, "tolist", forbid_host_copy)
    g_cum, out, a, final = owner._chunk_gated_delta_rule_fwd(**inputs)
    k, g, beta, kwargs = calls["kkt"]
    torch.testing.assert_close(k, inputs["k"].transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(g, g_cum.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        beta, inputs["beta"].transpose(1, 2).float(), rtol=0, atol=0
    )
    assert all(x.is_contiguous() for x in (k, g, beta, a, out))
    assert kwargs["cu_seqlens"] is inputs["cu_seqlens_list"]
    assert kwargs["chunk_indices"] is inputs["chunk_indices"]
    assert calls["cumsum"]["cu_seqlens"] is inputs["cu_seqlens"]
    assert calls["cumsum"]["chunk_indices_out"] is inputs["chunk_indices_out"]
    solve_input, solve_kwargs = calls["solve"]
    assert solve_kwargs["layout"] == ("ntd" if packed else "bnsd")
    assert solve_input.shape == (
        (2, 8, chunk_size) if packed else (1, 2, 8, chunk_size)
    )
    assert a.shape == (1, 2, 8, chunk_size) and a.dtype == dtype
    assert calls["wu"][3] is a
    assert calls["wu"][0] is k and calls["wu"][2] is beta
    assert calls["wu"][4]["g"] is g
    assert calls["state"]["g"] is g and calls["output"]["g"] is g
    torch.testing.assert_close(out, inputs["v"], rtol=0, atol=0)
    assert final is None
    assert owner._chunk_gated_delta_rule_bwd is backward
    assert owner.ChunkGatedDeltaRuleFunction is autograd
    assert owner.chunk_scaled_dot_kkt_fwd is triton


@pytest.mark.parametrize("unsupported", ["dtype", "mismatch", "chunk"])
def test_native_kkt_rejects_unsupported_contract(native_kkt_modules, unsupported):
    """Unsupported native inputs fail before launching any kernels."""
    owner, _, calls, inputs = native_kkt_modules
    if unsupported == "dtype":
        for name in ("q", "k", "v"):
            inputs[name] = inputs[name].double()
    elif unsupported == "mismatch":
        inputs["v"] = inputs["v"].half()
    else:
        inputs["chunk_size"] = 32
    assert ensure_mindspeed_gdn_native_kkt()
    with pytest.raises(ValueError, match="AscendC GDN requires"):
        owner._chunk_gated_delta_rule_fwd(**inputs)
    assert not calls


def test_native_kkt_propagates_kernel_failure(native_kkt_modules):
    """A native failure never falls back to the faulty Triton path."""
    owner, modules, _, inputs = native_kkt_modules

    def fail(*args, **kwargs):
        raise RuntimeError("native failure")

    modules["fla_npu.ops.ascendc"].npu_chunk_scaled_dot_kkt = fail
    assert ensure_mindspeed_gdn_native_kkt()
    with pytest.raises(RuntimeError, match="native failure"):
        owner._chunk_gated_delta_rule_fwd(**inputs)


def test_native_kkt_installation_is_idempotent(native_kkt_modules):
    """Repeated engine setup does not stack wrappers."""
    owner, _, _, _ = native_kkt_modules
    assert ensure_mindspeed_gdn_native_kkt()
    forward = owner._chunk_gated_delta_rule_fwd
    assert forward._areal_native_kkt
    assert ensure_mindspeed_gdn_native_kkt()
    assert owner._chunk_gated_delta_rule_fwd is forward


@pytest.mark.parametrize(
    "broken", ["native", "solve", "signature", "order", "parameter_kind"]
)
def test_native_kkt_incompatible_backend_preserves_bindings(native_kkt_modules, broken):
    """Validate the dependency contract before modifying its forward binding."""
    owner, modules, _, _ = native_kkt_modules
    if broken == "native":
        del modules["fla_npu.ops.ascendc"].npu_chunk_scaled_dot_kkt
    elif broken == "solve":
        owner.npu_solve_tri = None
    elif broken == "signature":
        owner._chunk_gated_delta_rule_fwd = lambda **kwargs: None
    else:
        import inspect

        params = list(
            inspect.signature(owner._chunk_gated_delta_rule_fwd).parameters.values()
        )
        if broken == "order":
            params[0], params[1] = params[1], params[0]
        else:
            params[-1] = params[-1].replace(kind=inspect.Parameter.KEYWORD_ONLY)
        owner._chunk_gated_delta_rule_fwd.__signature__ = inspect.Signature(params)
    before = vars(owner).copy()
    with pytest.raises(RuntimeError, match="backend is incompatible"):
        ensure_mindspeed_gdn_native_kkt()
    assert vars(owner) == before


def test_native_kkt_other_backend_preserves_bindings(native_kkt_modules):
    """Respect explicit selection of the non-FLA backend."""
    owner, modules, _, _ = native_kkt_modules
    before = vars(owner).copy()
    modules["mindspeed.core.ssm.gdn_backend"].load_gdn_operators = (
        lambda: types.SimpleNamespace(name="mindspeed")
    )
    assert not ensure_mindspeed_gdn_native_kkt()
    assert vars(owner) == before


def test_native_kkt_missing_backend_preserves_bindings(native_kkt_modules):
    """Keep older MindSpeed installations usable when the selector is absent."""
    owner, modules, _, _ = native_kkt_modules
    before = vars(owner).copy()
    del modules["mindspeed.core.ssm.gdn_backend"]
    assert not ensure_mindspeed_gdn_native_kkt()
    assert vars(owner) == before


def test_native_kkt_mindspeed_integration_preserves_bindings(native_kkt_modules):
    """Prefer a generalized upstream integration without layering a patch."""
    owner, modules, _, _ = native_kkt_modules
    owner._NATIVE_KKT = True
    before = vars(owner).copy()
    del modules["fla_npu.ops.ascendc"]
    assert ensure_mindspeed_gdn_native_kkt()
    assert vars(owner) == before


def test_native_kkt_upgrades_packed_only_integration(native_kkt_modules):
    """A prior packed-only marker must not bypass the generalized replacement."""
    owner, _, calls, inputs = native_kkt_modules
    owner._NATIVE_KKT_PACKED_BF16 = True
    original = owner._chunk_gated_delta_rule_fwd
    assert ensure_mindspeed_gdn_native_kkt()
    assert owner._chunk_gated_delta_rule_fwd is not original
    inputs.update(
        cu_seqlens=None,
        cu_seqlens_list=None,
        chunk_indices=None,
        chunk_indices_out=None,
    )
    owner._chunk_gated_delta_rule_fwd(**inputs)
    assert calls["solve"][1]["layout"] == "bnsd"
