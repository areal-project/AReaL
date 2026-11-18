# SPDX-License-Identifier: Apache-2.0
"""DeepGEMM receive-layout contracts; real kernel checks require CUDA."""

import sys
from types import SimpleNamespace

import pytest
import torch

from areal.engine.awex.qwen3_5_fp8 import Qwen35FP8Policy, Qwen35FP8Receiver


def _runtime(packed: bool, device: str = "cpu") -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace(num_attention_heads=8, num_key_value_heads=2)
    model.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((256, 1024), dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        ),
    )
    scale = (
        torch.zeros((2, 256), dtype=torch.int32, device=device).t()
        if packed
        else torch.ones((2, 8), dtype=torch.float32, device=device)
    )
    model.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
    )
    model.weight_scale_inv.format_ue8m0 = packed
    return model


@pytest.mark.parametrize("packed", [False, True])
def test_deepgemm_runtime_matches_selected_scale_format(monkeypatch, packed):
    """Accept each native layout, but reject a stale format flag."""
    model = _runtime(packed)
    policy = Qwen35FP8Policy(Qwen35FP8Policy.checkpoint_config())
    runner = object()
    model.orig_dtype = torch.bfloat16
    model.quant_method = SimpleNamespace(
        w8a8_block_fp8_linear=runner,
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers",
        SimpleNamespace(deep_gemm_wrapper=SimpleNamespace(DEEPGEMM_SCALE_UE8M0=packed)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.fp8_utils",
        SimpleNamespace(deepgemm_w8a8_block_fp8_linear_with_fallback=runner),
    )
    server = SimpleNamespace(fp8_gemm_runner_backend="deep_gemm")
    policy.validate_runtime(model, server)
    model.weight_scale_inv.format_ue8m0 = not packed
    with pytest.raises(ValueError, match="storage"):
        policy.validate_runtime(model, server)


def test_deepgemm_metadata_uses_plain_scales_without_changing_runtime():
    """Offloaded metadata discovery must not allocate CUDA staging or rebind scales."""
    model = _runtime(True)
    receiver = Qwen35FP8Receiver(model, {"weight"})
    metadata = dict(receiver.named_parameters(metadata_only=True))
    assert metadata["weight"] is model.weight
    assert metadata["weight_scale_inv"].device.type == "meta"
    assert metadata["weight_scale_inv"].dtype == torch.float32
    assert metadata["weight_scale_inv"].shape == (2, 8)
    assert model.weight_scale_inv.dtype == torch.int32
    assert model.weight_scale_inv.shape == (256, 2)
    assert next(receiver.parameters()) is model.weight
    assert receiver.config is model.config


def test_deepgemm_updates_refresh_weights_and_scales_at_stable_addresses(monkeypatch):
    """Two wire updates must refresh both runtime buffers and preserve cached views."""
    model = _runtime(True)
    receiver = Qwen35FP8Receiver(model, {"weight"})
    wire = dict(receiver.named_parameters())
    pointers = (model.weight.data_ptr(), model.weight_scale_inv.data_ptr())
    scale_stride = model.weight_scale_inv.stride()
    original_scale = model.weight_scale_inv
    completed = []

    def requant(weight, scale, block):
        assert block == [128, 128]
        return (
            (weight.float() + scale[0, 0]).to(torch.float8_e4m3fn),
            scale[0, 0].to(torch.int32).expand(256, 2).contiguous(),
        )

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.fp8_utils",
        SimpleNamespace(requant_weight_ue8m0=requant),
    )
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: completed.append(device)
    )
    for version in (1, 2):
        receiver.validate_storage()
        wire["weight"].data.copy_(torch.full((256, 1024), float(version)))
        wire["weight_scale_inv"].fill_(version * 3)
        receiver.finish_update()
        torch.testing.assert_close(
            model.weight.float(),
            torch.full((256, 1024), float(version * 4)),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.weight_scale_inv,
            torch.full((256, 2), version * 3, dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        assert model.weight_scale_inv is original_scale
        assert (model.weight.data_ptr(), model.weight_scale_inv.data_ptr()) == pointers
        assert model.weight_scale_inv.stride() == scale_stride
        assert (
            dict(receiver.named_parameters())["weight_scale_inv"]
            is wire["weight_scale_inv"]
        )
    assert len(completed) == 2


@pytest.mark.parametrize("replace_parameter", [False, True])
def test_deepgemm_replaced_storage_is_rejected_before_receive(replace_parameter):
    """Both Parameter replacement and data rebinding invalidate cached AWEX views."""
    model = _runtime(True)
    receiver = Qwen35FP8Receiver(model, {"weight"})
    with pytest.raises(RuntimeError, match="model changed"):
        receiver.validate_storage(_runtime(True))
    if replace_parameter:
        model.weight = torch.nn.Parameter(model.weight.clone(), requires_grad=False)
    else:
        model.weight.data = model.weight.clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        receiver.validate_storage()


def test_plain_fp32_scales_use_original_buffers_without_requantization(monkeypatch):
    """The existing Triton and FP32 DeepGEMM paths need no staging or SGLang imports."""
    model = _runtime(False)
    receiver = Qwen35FP8Receiver(model, {"weight"})
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.fp8_utils", None)
    assert (
        dict(receiver.named_parameters())["weight_scale_inv"] is model.weight_scale_inv
    )
    receiver.finish_update()


def test_deepgemm_staged_scales_preserve_native_awex_gate_up_views():
    """The native converter must split the plain grid into writable gate/up views."""
    awex = pytest.importorskip("awex.models.qwen3_5")
    layer = _runtime(True)
    receiver = Qwen35FP8Receiver(layer, {"weight"})
    converter = awex.SGlangToHFWeightConverterQwen3_5(
        layer.config,
        SimpleNamespace(tp_size=2, ep_size=1, device_backend="cpu"),
        SimpleNamespace(tp_rank=0, ep_rank=0),
    )
    wire_scale = dict(receiver.named_parameters())["weight_scale_inv"]
    wire_scale.zero_()
    views = dict(
        converter.convert_param(
            "model.layers.0.mlp.gate_up_proj.weight_scale_inv", wire_scale
        )
    )
    views["model.layers.0.mlp.up_proj.weight_scale_inv"].fill_(7)
    torch.testing.assert_close(wire_scale[0], torch.zeros(8), rtol=0, atol=0)
    torch.testing.assert_close(wire_scale[1], torch.full((8,), 7.0), rtol=0, atol=0)
    torch.testing.assert_close(
        layer.weight_scale_inv, torch.zeros((256, 2), dtype=torch.int32), rtol=0, atol=0
    )


def test_deepgemm_frozen_mtp_scales_are_not_staged_or_converted(monkeypatch):
    """AWEX-filtered MTP weights must remain frozen even when stored as packed FP8."""
    model = _runtime(False)
    model.add_module("mtp", _runtime(True))
    original = model.mtp.weight_scale_inv.clone()
    receiver = Qwen35FP8Receiver(model, {"weight"})
    wire = dict(receiver.named_parameters())
    assert wire["mtp.weight_scale_inv"] is model.mtp.weight_scale_inv
    monkeypatch.setitem(sys.modules, "sglang.srt.layers.quantization.fp8_utils", None)
    receiver.finish_update()
    torch.testing.assert_close(model.mtp.weight_scale_inv, original, rtol=0, atol=0)


def test_deepgemm_adapter_metadata_and_export_use_matching_scale_views(
    monkeypatch, tmp_path
):
    """Metadata describes wire FP32, while snapshots unpack the current runtime scale."""
    awex = pytest.importorskip("awex.models.qwen3_5")
    from areal.engine.awex.adapters import sglang_adapter as adapter_module

    layer = _runtime(True)
    model = torch.nn.Module()
    model.config = layer.config
    model.add_module("model", torch.nn.Module())
    model.model.add_module("layers", torch.nn.ModuleList([torch.nn.Module()]))
    model.model.layers[0].add_module("mlp", torch.nn.Module())
    model.model.layers[0].mlp.add_module("gate_up_proj", layer)
    weight_name = "model.layers.0.mlp.gate_up_proj.weight"
    rank = SimpleNamespace(tp_rank=0, ep_rank=0, engine_rank=3)
    converter = awex.SGlangToHFWeightConverterQwen3_5(
        layer.config, SimpleNamespace(tp_size=1, ep_size=1, device_backend="cpu"), rank
    )
    instance = adapter_module.AwexSGLangAdapter(SimpleNamespace())
    instance._get_model = lambda: model
    instance._get_model_arch_name = lambda: "Qwen3_5ForConditionalGeneration"
    instance._get_model_context = lambda: {}
    instance._get_weight_converter = lambda rank_info: converter
    instance._rank_info = rank
    instance._parameter_layout = "awex"
    instance._legacy_engine_rank = 3
    instance._qwen35_fp8_receiver = Qwen35FP8Receiver(model, {weight_name})

    def extract_rank(context, engine_rank):
        assert engine_rank == 3
        return rank

    monkeypatch.setattr(adapter_module, "get_sglang_rank_info", extract_rank)
    metadata = instance._compute_legacy_local_raw_meta()
    assert metadata["rank_info"] is rank
    scale_name = "model.layers.0.mlp.up_proj.weight_scale_inv"
    scale_meta = next(p for p in metadata["params_meta"] if p["name"] == scale_name)
    assert (scale_meta["shape"], scale_meta["dtype"]) == ((1, 8), "float32")
    wire = instance.get_local_shard_parameters()
    wire[scale_name].fill_(3)

    def unpack(scale, shape, block):
        assert scale is layer.weight_scale_inv
        assert shape == (256, 1024) and block == [128, 128]
        return torch.full((2, 8), 7.0)

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.layers.quantization.fp8_utils",
        SimpleNamespace(_unpack_ue8m0_scale_for_triton=unpack),
    )
    path = tmp_path / "runtime.pt"
    instance.save_parameters(str(path), [scale_name])
    snapshot = torch.load(path, weights_only=True)
    torch.testing.assert_close(
        snapshot[scale_name], torch.full((1, 8), 7.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        wire[scale_name], torch.full((1, 8), 3.0), rtol=0, atol=0
    )
    assert instance._parameters is wire


def test_deepgemm_legacy_conversion_completes_after_receive(monkeypatch):
    """Every version must convert before update_weights returns to the ready signal."""
    pytest.importorskip("awex.models.qwen3_5")
    from areal.engine.awex.adapters import sglang_adapter as adapter_module

    events = []
    model = _runtime(False)
    instance = adapter_module.AwexSGLangAdapter(
        SimpleNamespace(
            server_args=SimpleNamespace(fp8_gemm_runner_backend="deep_gemm")
        )
    )
    instance._initialized = True
    instance._get_model = lambda: model
    instance._qwen35_fp8_receiver = SimpleNamespace(
        validate_storage=lambda current: events.append("validate"),
        finish_update=lambda: events.append("convert"),
    )
    instance._ensure_reader = lambda: SimpleNamespace(
        update_weights=lambda step_id: events.append(("receive", step_id)),
    )
    monkeypatch.setattr(adapter_module, "_reapply_aiter_moe_layout", lambda model: 0)
    for version in (1, 2):
        instance.update_weights(version)
        events.append("ready")
    assert events == [
        "validate",
        ("receive", 1),
        "convert",
        "ready",
        "validate",
        ("receive", 2),
        "convert",
        "ready",
    ]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="DeepGEMM requires CUDA")
def test_deepgemm_repeated_updates_match_fresh_load_with_cuda_graph():
    """Real quantization/GEMM: repeated wire updates match fresh load after graph replay."""
    pytest.importorskip("deep_gemm")
    fp8 = pytest.importorskip("sglang.srt.layers.quantization.fp8_utils")
    from sglang.srt.layers import deep_gemm_wrapper

    from areal.engine.megatron_utils.fp8.kernels import blockwise_cast_to_fp8_triton

    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        pytest.skip("SGLang DeepGEMM runner is unavailable on this GPU")
    packed = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
    model = _runtime(packed, "cuda")
    model.orig_dtype = torch.bfloat16
    model.quant_method = SimpleNamespace(
        w8a8_block_fp8_linear=fp8.deepgemm_w8a8_block_fp8_linear_with_fallback,
        quant_config=SimpleNamespace(weight_block_size=[128, 128]),
    )
    initial = torch.zeros_like(model.weight, dtype=torch.bfloat16)
    initial_weight, initial_scale = blockwise_cast_to_fp8_triton(initial, (128, 128))
    if packed:
        initial_weight, initial_scale = fp8.requant_weight_ue8m0(
            initial_weight, initial_scale, [128, 128]
        )
    model.weight.data.copy_(initial_weight)
    model.weight_scale_inv.data.copy_(initial_scale)
    policy = Qwen35FP8Policy(Qwen35FP8Policy.checkpoint_config())
    policy.validate_runtime(model, SimpleNamespace(fp8_gemm_runner_backend="deep_gemm"))
    receiver = Qwen35FP8Receiver(model, {"weight"})
    wire = dict(receiver.named_parameters())
    x = torch.randn((16, 1024), dtype=torch.bfloat16, device="cuda")

    def forward(weight, scale):
        return fp8.deepgemm_w8a8_block_fp8_linear_with_fallback(
            x, weight, [128, 128], scale
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward(model.weight, model.weight_scale_inv)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = forward(model.weight, model.weight_scale_inv)
    pointers = (model.weight.data_ptr(), model.weight_scale_inv.data_ptr())
    for version in (1, 2):
        source = torch.randn_like(initial) * (0.01 * version)
        q, scale = blockwise_cast_to_fp8_triton(source, (128, 128))
        expected_q, expected_scale = (
            fp8.requant_weight_ue8m0(q, scale, [128, 128]) if packed else (q, scale)
        )
        receiver.validate_storage()
        wire["weight"].data.copy_(q)
        wire["weight_scale_inv"].copy_(scale)
        receiver.finish_update()
        graph.replay()
        torch.testing.assert_close(
            result, forward(expected_q, expected_scale), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            model.weight.float(), expected_q.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.weight_scale_inv, expected_scale, rtol=0, atol=0
        )
        assert (model.weight.data_ptr(), model.weight_scale_inv.data_ptr()) == pointers
