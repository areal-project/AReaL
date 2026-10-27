# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 FP8 training precision and inference configuration contracts."""

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import FP8EngineConfig, SGLangConfig


def test_fp8_provider_mapping_keeps_export_options_local():
    config = FP8EngineConfig(recipe="blockwise", param=False, direct_convert=False)
    provider = SimpleNamespace(
        fp8=None, fp8_recipe=None, fp8_param=True, tp_only_amax_red=True
    )
    config.apply_to(provider)
    assert (provider.fp8, provider.fp8_recipe, provider.fp8_param) == (
        "e4m3",
        "blockwise",
        False,
    )
    assert provider.tp_only_amax_red is False
    assert not hasattr(provider, "fp8_direct_convert")
    assert SGLangConfig().fp8_gemm_runner_backend is None


def test_dense_fp8_training_requires_unquantized_bf16_storage():
    module = pytest.importorskip("areal.engine.megatron_engine")
    engine = SimpleNamespace(
        enable_fp8=True,
        bridge_cls="megatron-bridge",
        hf_config=SimpleNamespace(model_type="qwen3_5"),
        dtype=torch.bfloat16,
        fp8_config=FP8EngineConfig(param=False),
        mcore_config=SimpleNamespace(ddp=SimpleNamespace(fp8_param_gather=False)),
        quantization_config=None,
    )
    module.MegatronEngine._validate_fp8_consistency(engine)
    engine.fp8_config.param = True
    with pytest.raises(ValueError, match="BF16 parameter storage"):
        module.MegatronEngine._validate_fp8_consistency(engine)
    engine.fp8_config.param = False
    engine.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="BF16 Qwen3.5 checkpoint"):
        module.MegatronEngine._validate_fp8_consistency(engine)


def test_sglang_fp8_backend_uses_upstream_cli_spelling(monkeypatch):
    monkeypatch.setattr(
        "areal.api.cli_args.pkg_version.is_version_greater_or_equal", lambda *args: True
    )
    defaults = SGLangConfig.build_args(SGLangConfig(), tp_size=1, base_gpu_id=0)
    assert "fp8_gemm_runner_backend" not in defaults
    args = SGLangConfig.build_args(
        SGLangConfig(fp8_gemm_runner_backend="triton"), tp_size=1, base_gpu_id=0
    )
    command = SGLangConfig.build_cmd_from_args(args)
    assert command[command.index("--fp8-gemm-backend") + 1] == "triton"
    assert "--fp8-gemm-runner-backend" not in command
    assert args["fp8_gemm_runner_backend"] == "triton"


def test_gdn_native_recipe_disables_fp8_only_for_fused_input_projection():
    te = pytest.importorskip("megatron.core.extensions.transformer_engine")
    if not te.HAVE_TE:
        pytest.skip("TransformerEngine is not installed")
    gdn = pytest.importorskip("megatron.core.ssm.gated_delta_net.gdn")
    from areal.engine.megatron_utils.qwen3_5_fp8 import Qwen35FP8Training

    # Construct module shells: this checks native recipe/autocast selection
    # without allocating a CUDA model or mocking distributed tensor internals.
    layer = gdn.GatedDeltaNet.__new__(gdn.GatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.layer_number = 1
    in_proj = te.TELayerNormColumnParallelLinear.__new__(
        te.TELayerNormColumnParallelLinear
    )
    torch.nn.Module.__init__(in_proj)
    layer.in_proj = in_proj
    layer.out_proj = torch.nn.Identity()
    models = [layer]
    assert Qwen35FP8Training.prepare_model(models) is models
    assert not in_proj.will_execute_quantized(is_context_quantized=True)
    in_proj.eval()
    assert not in_proj.will_execute_quantized(is_context_quantized=True)
    assert not hasattr(layer.out_proj, "te_quant_params")
