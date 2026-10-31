# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 FP8 conversion and AWEX contracts; GPU cases skip without CUDA."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import FP8EngineConfig, SGLangConfig
from areal.engine.awex.qwen3_5_fp8 import Qwen35FP8Policy


@pytest.fixture
def policy():
    return Qwen35FP8Policy(Qwen35FP8Policy.checkpoint_config())


@pytest.mark.parametrize(
    ("name", "quantized"),
    [
        ("model.language_model.layers.0.self_attn.q_proj.weight", True),
        ("model.layers.0.self_attn.qkv_proj.weight", True),
        ("model.layers.0.linear_attn.in_proj_qkvz.weight", True),
        ("model.layers.0.linear_attn.out_proj.weight", True),
        ("model.layers.0.mlp.gate_proj.weight", True),
        ("model.layers.0.mlp.down_proj.weight", True),
        ("model.layers.0.linear_attn.in_proj_ba.weight", False),
        ("model.layers.0.linear_attn.conv1d.weight", False),
        ("model.layers.0.linear_attn.A_log", False),
        ("model.layers.0.mlp.gate.weight", False),
        ("model.visual.blocks.0.attn.proj.weight", False),
        ("lm_head.weight", False),
    ],
)
def test_projection_policy_keeps_sensitive_layers_in_bf16(policy, name, quantized):
    assert policy.quantizes(name) is quantized


def test_fused_projection_partial_ignore_is_rejected():
    policy = Qwen35FP8Policy(
        Qwen35FP8Policy.checkpoint_config(["layers.0.self_attn.q_proj"])
    )
    with pytest.raises(ValueError, match="same precision"):
        policy.quantizes("model.layers.0.self_attn.qkv_proj.weight")


@pytest.mark.parametrize(
    "override",
    [
        {"weight_block_size": [64, 128]},
        {"scale_fmt": "ue8m0"},
        {"activation_scheme": "static"},
    ],
)
def test_unsupported_scale_contract_is_rejected(override):
    with pytest.raises(ValueError, match="unpacked FP32 scales"):
        Qwen35FP8Policy(Qwen35FP8Policy.checkpoint_config() | override)


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


def test_pipeline_parallelism_rejects_layer_specific_projection_exclusions(policy):
    policy.validate_pipeline_exclusions(pp_size=2)
    policy = Qwen35FP8Policy(
        Qwen35FP8Policy.checkpoint_config(
            [
                "model.language_model.layers.20.mlp.down_proj",
            ]
        )
    )
    policy.validate_pipeline_exclusions(pp_size=1)
    with pytest.raises(ValueError, match="uniform projection exclusions"):
        policy.validate_pipeline_exclusions(pp_size=2)


def _metadata():
    def param(name, dtype, shape, offset):
        shard = SimpleNamespace(
            engine_rank=0,
            global_rank=1,
            shape=shape,
            global_offset=offset,
            sharding_dim=1,
            sharding_type="TP",
        )
        return SimpleNamespace(
            name=name,
            dtype=dtype,
            global_shape=(shape[0], shape[1] * 2),
            shards=[shard],
        )

    name = "model.layers.0.linear_attn.out_proj.weight"
    return [
        param(name, torch.float8_e4m3fn, (256, 128), (0, 128)),
        param(name + "_scale_inv", torch.float32, (2, 1), (0, 1)),
    ]


def test_scale_metadata_tracks_column_shards():
    metadata = _metadata()
    Qwen35FP8Policy.validate_metadata(metadata)
    bad = deepcopy(metadata)
    bad[1].shards[0].global_offset = (1, 0)
    with pytest.raises(ValueError, match="scale sharding"):
        Qwen35FP8Policy.validate_metadata(bad)
    with pytest.raises(ValueError, match="Missing FP32 scale"):
        Qwen35FP8Policy.validate_metadata(metadata[:1])


def test_runtime_requires_triton_and_writable_fp32_scale_grid(policy):
    model = torch.nn.Module()
    model.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((256, 128), dtype=torch.float8_e4m3fn), requires_grad=False
        ),
    )
    model.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.ones((2, 1), dtype=torch.float32), requires_grad=False
        ),
    )
    server = SimpleNamespace(fp8_gemm_runner_backend="auto")
    with pytest.raises(ValueError, match="backend=triton"):
        policy.validate_runtime(model, server)
    server.fp8_gemm_runner_backend = "triton"
    policy.validate_runtime(model, server)
    model.weight_scale_inv.format_ue8m0 = True
    with pytest.raises(ValueError, match="FP32 block grid"):
        policy.validate_runtime(model, server)


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


def test_fused_gdn_components_require_individual_block_alignment():
    # Total fused rows are aligned, but the individual Q/K slices are not.
    config = dict(
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=1024,
        head_dim=128,
        linear_num_key_heads=1,
        linear_key_head_dim=128,
        linear_num_value_heads=4,
        linear_value_head_dim=128,
    )
    with pytest.raises(ValueError, match="GDN query/key"):
        Qwen35FP8Policy.validate_parallelism(config, infer_tp=2)


def test_replicated_kv_heads_allow_wider_inference_tp():
    config = dict(
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_size=4096,
        head_dim=256,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=32,
        linear_value_head_dim=128,
    )
    Qwen35FP8Policy.validate_parallelism(config, infer_tp=8)


def test_awex_scale_views_and_sharding_are_write_through(monkeypatch):
    pytest.importorskip("awex.models.qwen3_5")
    from awex.models.registry import ModelRegistry, get_infer_weights_converter
    from awex.sharding.sglang_sharding import get_sglang_sharding_strategy

    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    Qwen35FP8Policy.register_awex()
    Qwen35FP8Policy.register_awex()
    config = SimpleNamespace(
        num_attention_heads=8, num_key_value_heads=2, hidden_size=1024
    )
    rank = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        attn_tp_size=2,
        attn_tp_rank=0,
        ep_size=1,
        ep_rank=0,
        ep_tp_size=2,
        ep_tp_rank=0,
        pp_size=1,
        pp_rank=0,
    )
    server = SimpleNamespace(
        tp_size=2,
        ep_size=1,
        device_backend="cpu",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=None,
    )
    converter = get_infer_weights_converter(
        "sglang", "Qwen3_5ForConditionalGeneration", config, rank, server
    )
    scales = torch.zeros((4, 2), dtype=torch.float32)
    views = dict(
        converter.convert_param(
            "model.layers.0.mlp.gate_up_proj.weight_scale_inv", scales
        )
    )
    views["model.layers.0.mlp.up_proj.weight_scale_inv"].fill_(3)
    torch.testing.assert_close(scales[2:], torch.full((2, 2), 3.0), rtol=0, atol=0)
    torch.testing.assert_close(scales[:2], torch.zeros((2, 2)), rtol=0, atol=0)
    strategy = get_sglang_sharding_strategy(
        "Qwen3_5ForConditionalGeneration", server, rank
    )
    weight = strategy.get_sharding_strategy(
        "model.layers.0.linear_attn.out_proj.weight"
    )
    scale = strategy.get_sharding_strategy(
        "model.layers.0.linear_attn.out_proj.weight_scale_inv"
    )
    assert weight == scale
    assert scale[1] == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FP8 Triton quantization requires CUDA"
)
def test_fp8_update_refreshes_weight_and_scale_without_nan(policy):
    name = "model.layers.0.linear_attn.out_proj.weight"
    source = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    source[:128, :128] = 0
    first = dict(policy.quantize(name, source))
    second = dict(policy.quantize(name, source * 2))
    scale = second[name + "_scale_inv"]
    assert scale.dtype == torch.float32
    assert torch.isfinite(scale).all()
    reconstructed = second[name].float() * scale.repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(
        reconstructed[:128, :128],
        torch.zeros((128, 128), device="cuda"),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(reconstructed, source.float() * 2, rtol=0.065, atol=0.01)
    torch.testing.assert_close(
        scale[1:, 1:], first[name + "_scale_inv"][1:, 1:] * 2, rtol=1e-6, atol=0
    )
