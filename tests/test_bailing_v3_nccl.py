# SPDX-License-Identifier: Apache-2.0

"""CPU checks for V3 TP gather, HF conversion, and NCCL sender buckets.

Tensor conversion uses the real mbridge and Megatron Core implementations. CUDA
FP8 imports are isolated when unavailable; no model or GPU kernel is mocked.
The sender methods are loaded independently of the optimizer/backend imports so
these tests also run without megatron-bridge or CUDA on a developer machine.
"""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from areal.engine.megatron_utils.bailing_v3 import (
    BailingV3MlaWeightPairs,
    is_bailing_v3,
    validate_bailing_v3_weight_update,
)


def _config(**overrides):
    values = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        model_type="bailing_hybrid",
        num_hidden_layers=8,
        hidden_size=16,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=2,
        kv_channels=2,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        no_kda_lora=True,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_shared_experts=1,
        num_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        layer_group_size=4,
        vocab_size=10,
        max_position_embeddings=1024,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def converter(monkeypatch):
    pytest.importorskip("megatron.core")
    if importlib.util.find_spec("triton") is None:
        # BF16 conversion never calls these optional CUDA quantization routines.
        fp8 = types.ModuleType("areal.engine.megatron_utils.fp8")
        fp8.FP8BlockwiseTensorHelper = type("FP8BlockwiseTensorHelper", (), {})

        def unexpected_fp8(*_args, **_kwargs):
            pytest.fail("BF16 conversion entered an FP8 routine")

        fp8.convert_fp8_helper_to_pytorch_fp8 = unexpected_fp8
        fp8.get_block_size_from_config = unexpected_fp8
        fp8.quantize_params = unexpected_fp8
        monkeypatch.setitem(sys.modules, fp8.__name__, fp8)

    if importlib.util.find_spec("megatron.bridge") is None:
        lora = types.ModuleType("areal.engine.megatron_utils.megatron_lora")

        def unexpected_lora(*_args, **_kwargs):
            pytest.fail("Full-model V3 conversion entered a LoRA routine")

        lora.convert_qwen3_lora_to_hf = unexpected_lora
        lora.convert_qwen3_moe_lora_to_hf = unexpected_lora
        monkeypatch.setitem(sys.modules, lora.__name__, lora)

    path = (
        Path(__file__).resolve().parents[1] / "areal/engine/megatron_utils/megatron.py"
    )
    spec = importlib.util.spec_from_file_location("_bailing_v3_nccl_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bridge_factory():
    pytest.importorskip("mbridge")
    from mbridge.core.parallel_states import ParallelStates

    from areal.models.mcore.bailing_v3_bridge import BailingV3Bridge

    def make(config):
        return BailingV3Bridge(config, parallel_states=ParallelStates(vpp_size=None))

    return make


def _gather(converter, monkeypatch, name, shards, dim=0, *, glu=False):
    def all_gather(outputs, tensor, *, group):
        assert group == "tp"
        for output, shard in zip(outputs, shards, strict=True):
            output.copy_(shard)

    monkeypatch.setattr(converter.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        converter.mpu, "get_tensor_model_parallel_world_size", lambda: len(shards)
    )
    monkeypatch.setattr(converter.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(
        converter.mpu, "get_expert_tensor_parallel_world_size", lambda: len(shards)
    )
    monkeypatch.setattr(converter.mpu, "get_expert_tensor_parallel_group", lambda: "tp")
    param = torch.nn.Parameter(shards[0])
    param.tensor_model_parallel = True
    param.partition_dim = dim
    param.partition_stride = 1
    return converter.all_gather_param(name, param, gated_linear_unit=glu)


def _convert(converter, bridge, name, tensor, **kwargs):
    return converter.convert_to_hf(
        bridge.config,
        bridge.hf_config.model_type,
        name,
        tensor,
        hf_config=bridge.hf_config,
        bridge=bridge,
        **kwargs,
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "kind,components,tail_shape",
    [
        ("in_proj", ["q_proj", "k_proj", "v_proj", "f_proj", "g_proj"], (16,)),
        ("conv1d", ["q_conv1d", "k_conv1d", "v_conv1d"], (1, 4)),
    ],
)
def test_kda_gather_reconstructs_each_full_hf_projection(
    converter, bridge_factory, monkeypatch, tp_size, kind, components, tail_shape
):
    bridge = bridge_factory(_config())
    full = [
        torch.arange(16 * torch.Size(tail_shape).numel(), dtype=torch.float32).reshape(
            16, *tail_shape
        )
        + 1000 * index
        for index in range(len(components))
    ]
    # Local layout is [q_rank, k_rank, v_rank, f_rank, g_rank], not
    # a contiguous slice of [q_full, k_full, v_full, f_full, g_full].
    shards = [
        torch.cat([tensor.chunk(tp_size, dim=0)[rank] for tensor in full])
        for rank in range(tp_size)
    ]
    name = f"module.module.decoder.layers.0.self_attention.{kind}.weight"
    gathered = _gather(converter, monkeypatch, name, shards)
    converted = _convert(converter, bridge, name, gathered)

    assert [name for name, _ in converted] == [
        f"model.layers.0.attention.{component}.weight" for component in components
    ]
    for (_, actual), expected in zip(converted, full, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.is_contiguous()


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize(
    "suffix",
    [
        "mlp.linear_fc1.weight",
        "mlp.shared_experts.linear_fc1.weight",
        "mlp.experts.linear_fc1.weight7",
    ],
)
def test_glu_gather_is_deinterleaved_once(
    converter, bridge_factory, monkeypatch, tp_size, suffix
):
    bridge = bridge_factory(_config())
    gate = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16)
    up = gate + 1000
    shards = [
        torch.cat([gate.chunk(tp_size)[rank], up.chunk(tp_size)[rank]])
        for rank in range(tp_size)
    ]
    name = f"module.module.decoder.layers.4.{suffix}"
    gathered = _gather(converter, monkeypatch, name, shards, glu=True)
    converted = _convert(converter, bridge, name, gathered)
    hf_prefix = "model.layers.4.mlp"
    if "shared_experts" in suffix:
        hf_prefix += ".shared_experts"
    elif ".experts." in suffix:
        hf_prefix += ".experts.7"
    assert [key for key, _ in converted] == [
        f"{hf_prefix}.gate_proj.weight",
        f"{hf_prefix}.up_proj.weight",
    ]
    torch.testing.assert_close(converted[0][1], gate, rtol=0, atol=0)
    torch.testing.assert_close(converted[1][1], up, rtol=0, atol=0)


@pytest.mark.parametrize("q_lora_rank", [None, 4])
@pytest.mark.parametrize(
    "suffix,hf_suffix",
    [
        ("input_layernorm.weight", "input_layernorm.weight"),
        ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
        ("self_attention.linear_gate.weight", "attention.g_proj.weight"),
        (
            "self_attention.linear_kv_down_proj.weight",
            "attention.kv_a_proj_with_mqa.weight",
        ),
        ("self_attention.linear_kv_up_proj.weight", "attention.kv_b_proj.weight"),
        (
            "self_attention.linear_kv_up_proj.layer_norm_weight",
            "attention.kv_a_layernorm.weight",
        ),
        ("self_attention.linear_proj.weight", "attention.dense.weight"),
        ("mlp.router.weight", "mlp.gate.weight"),
        ("mlp.router.expert_bias", "mlp.gate.expert_bias"),
        ("mlp.shared_experts.linear_fc2.weight", "mlp.shared_experts.down_proj.weight"),
        ("mlp.experts.linear_fc2.weight6", "mlp.experts.6.down_proj.weight"),
    ],
)
def test_mla_and_moe_names_share_the_hf_save_mapping(
    converter, bridge_factory, q_lora_rank, suffix, hf_suffix
):
    bridge = bridge_factory(_config(q_lora_rank=q_lora_rank))
    tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    converted = _convert(
        converter, bridge, f"module.module.decoder.layers.3.{suffix}", tensor
    )
    assert len(converted) == 1
    assert converted[0][0] == f"model.layers.3.{hf_suffix}"
    torch.testing.assert_close(converted[0][1], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("q_lora_rank", [None, 4])
def test_mla_q_projection_layout_follows_config(converter, bridge_factory, q_lora_rank):
    bridge = bridge_factory(_config(q_lora_rank=q_lora_rank))
    mapping = (
        {"linear_q_proj.weight": "q_proj.weight"}
        if q_lora_rank is None
        else {
            "linear_q_down_proj.weight": "q_a_proj.weight",
            "linear_q_up_proj.weight": "q_b_proj.weight",
            "linear_q_up_proj.layer_norm_weight": "q_a_layernorm.weight",
        }
    )
    for suffix, hf_suffix in mapping.items():
        tensor = torch.ones(4, 16)
        converted = _convert(
            converter,
            bridge,
            f"module.module.decoder.layers.3.self_attention.{suffix}",
            tensor,
        )
        assert converted[0][0] == f"model.layers.3.attention.{hf_suffix}"
        torch.testing.assert_close(converted[0][1], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("suffix,rows", [("A_log", 8), ("dt_bias", 16)])
def test_kda_decay_parameters_keep_full_fp32_values(
    converter, bridge_factory, monkeypatch, tp_size, suffix, rows
):
    tensor = torch.linspace(-0.1234567, 0.9876543, rows, dtype=torch.float32)
    name = f"module.module.decoder.layers.0.self_attention.{suffix}"
    gathered = _gather(converter, monkeypatch, name, tensor.chunk(tp_size))
    converted = _convert(converter, bridge_factory(_config()), name, gathered)
    assert converted[0][0] == f"model.layers.0.attention.{suffix}"
    assert converted[0][1].dtype == torch.float32
    torch.testing.assert_close(converted[0][1], tensor, rtol=0, atol=0)


def test_hybrid_v25_dispatch_remains_unchanged(converter):
    config = _config(architectures=["BailingMoeV2_5ForCausalLM"])
    tensor = torch.arange(8 * 3 * 2 * 16, dtype=torch.float32).reshape(48, 16)
    converted = converter.convert_to_hf(
        config,
        "bailing_hybrid",
        "module.module.decoder.layers.0.self_attention.linear_qkv.weight",
        tensor,
        hf_config=config,
    )
    expected = torch.cat(
        [tensor.reshape(8, 3, 2, 16)[:, i].reshape(16, 16) for i in range(3)]
    )
    assert converted[0][0] == "model.layers.0.attention.query_key_value.weight"
    torch.testing.assert_close(converted[0][1], expected, rtol=0, atol=0)


def test_v3_requires_explicit_bridge_instead_of_v25_dispatch(converter):
    with pytest.raises(ValueError, match="requires its model bridge"):
        converter.convert_to_hf(
            _config(),
            "bailing_hybrid",
            "module.module.decoder.layers.0.self_attention.in_proj.weight",
            torch.empty(80, 16),
            hf_config=_config(),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"use_lora": True},
        {"quantization_config": {"quant_method": "fp8"}},
        {"fp8_direct_convert": True},
    ],
)
def test_unsupported_precision_modes_fail_before_weight_transfer(options):
    with pytest.raises(NotImplementedError, match="unquantized full-model"):
        validate_bailing_v3_weight_update(_config(), **options)


def test_kda_lora_layout_is_rejected():
    with pytest.raises(NotImplementedError, match="no_kda_lora=True"):
        validate_bailing_v3_weight_update(_config(no_kda_lora=False))


def _sender_methods(converter):
    """Execute production methods while excluding unrelated CUDA backend imports."""
    path = Path(__file__).resolve().parents[1] / "areal/engine/megatron_engine.py"
    tree = ast.parse(path.read_text())
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine"
    )
    names = {
        "_collect_param",
        "_impl_update_weight_from_distributed",
        "_update_weights_via_registry",
        "_update_bucket_expert_weights_from_distributed",
        "_impl_update_expert_weight_from_distributed",
        "_init_weight_update_from_distributed",
    }
    methods = [
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    normalizer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_normalize_glu_param_name"
    )
    normalizer_constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id in {"_LAYER_IDX_RE", "_EXPERT_NUM_RE"}
            for target in node.targets
        )
    ]
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        re=__import__("re"),
        dist=converter.dist,
        mpu=converter.mpu,
        BailingV3MlaWeightPairs=BailingV3MlaWeightPairs,
        is_bailing_v3=is_bailing_v3,
        validate_bailing_v3_weight_update=validate_bailing_v3_weight_update,
        convert_to_hf=converter.convert_to_hf,
        all_gather_param=converter.all_gather_param,
        remove_padding=converter.remove_padding,
        get_named_parameters=converter.get_named_parameters,
        lang_config=lambda config: config,
        FP8BlockwiseTensorHelper=converter.FP8BlockwiseTensorHelper,
        current_platform=SimpleNamespace(current_device=lambda: "cpu"),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *normalizer_constants,
            normalizer,
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return type("SenderMethods", (), {name: namespace[name] for name in names})


@pytest.mark.parametrize("q_lora_rank", [None, 4])
@pytest.mark.parametrize("bucket_bytes", [1, 256, 1024])
def test_sender_keeps_lowrank_mla_pairs_in_one_native_loader_call(
    converter, bridge_factory, monkeypatch, q_lora_rank, bucket_bytes
):
    bridge = bridge_factory(_config(q_lora_rank=q_lora_rank))
    sender = _sender_methods(converter)()
    sender.bridge = bridge
    sender.hf_config = bridge.hf_config
    sender.tf_config = bridge.config
    sender.config = SimpleNamespace(use_lora=False)
    sender.quantization_config = None
    sender.fp8_direct_convert = False
    sender._glu_fc1_names = set()
    sender._duplicated_param_names = set()
    sender.cpu_group = "cpu"
    sender.is_pipeline_parallel_head = lambda: True
    tensors = []
    expected = {}
    for layer in (3, 7):
        mapping = [
            (
                "linear_q_proj" if q_lora_rank is None else "linear_q_down_proj",
                "q_proj" if q_lora_rank is None else "q_a_proj",
            ),
            ("linear_gate", "g_proj"),
            ("linear_kv_down_proj", "kv_a_proj_with_mqa"),
        ]
        for index, (mcore_name, hf_name) in enumerate(mapping):
            tensor = torch.full((4, 16), layer + index, dtype=torch.bfloat16)
            tensors.append(
                (
                    f"module.module.decoder.layers.{layer}.self_attention.{mcore_name}.weight",
                    tensor,
                )
            )
            expected[f"model.layers.{layer}.attention.{hf_name}.weight"] = tensor
    sender.model = tensors
    method_globals = sender._update_weights_via_registry.__func__.__globals__
    monkeypatch.setitem(
        method_globals, "get_named_parameters", lambda model, _experts: iter(model)
    )
    monkeypatch.setattr(converter.dist, "barrier", lambda **_kwargs: None)
    buckets = []

    def flush(_meta, weights):
        if weights:
            buckets.append(dict(weights))
        weights.clear()

    sender._update_bucket_weights_from_distributed = flush
    sender._update_weights_via_registry(
        SimpleNamespace(weight_chunked_mem_mb=bucket_bytes / (1024 * 1024))
    )
    received = {}
    for bucket in buckets:
        for layer in (3, 7):
            q_name = f"model.layers.{layer}.attention.q_a_proj.weight"
            kv_name = f"model.layers.{layer}.attention.kv_a_proj_with_mqa.weight"
            if q_lora_rank is not None:
                assert (q_name in bucket) == (kv_name in bucket)
        received.update(bucket)
    assert received.keys() == expected.keys()
    for name, tensor in received.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_pair_buffer_rejects_incomplete_or_interleaved_layers():
    pairs = BailingV3MlaWeightPairs()
    assert (
        pairs.group([("model.layers.3.attention.q_a_proj.weight", torch.ones(2, 2))])
        == []
    )
    with pytest.raises(RuntimeError, match="Incomplete"):
        pairs.finish()
    with pytest.raises(RuntimeError, match="before the next pair"):
        pairs.group([("model.layers.7.attention.q_a_proj.weight", torch.ones(2, 2))])


def test_global_pp_layers_and_ep_experts_reach_the_v3_bridge(
    converter, bridge_factory, monkeypatch
):
    monkeypatch.setattr(
        converter.mpu, "get_expert_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(converter.mpu, "get_expert_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        converter.mpu, "get_virtual_pipeline_model_parallel_rank", lambda: None
    )
    monkeypatch.setattr(
        converter, "get_transformer_layer_offset", lambda *_args, **_kwargs: 4
    )
    weight = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(4, 16))
    bias = torch.arange(8, dtype=torch.float32)
    model = SimpleNamespace(
        config=_config(),
        named_parameters=lambda: iter(
            [("module.module.decoder.layers.0.mlp.experts.linear_fc2.weight1", weight)]
        ),
        named_buffers=lambda: iter(
            [("module.module.decoder.layers.0.mlp.router.expert_bias", bias)]
        ),
    )
    bridge = bridge_factory(_config())
    converted = dict(
        result
        for name, tensor in converter.get_named_parameters(model, num_experts=8)
        for result in _convert(converter, bridge, name, tensor)
    )
    assert converted.keys() == {
        "model.layers.4.mlp.experts.5.down_proj.weight",
        "model.layers.4.mlp.gate.expert_bias",
    }
    torch.testing.assert_close(
        converted["model.layers.4.mlp.experts.5.down_proj.weight"],
        weight,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        converted["model.layers.4.mlp.gate.expert_bias"], bias, rtol=0, atol=0
    )


@pytest.mark.parametrize("projection", ["linear_q_down_proj", "linear_kv_down_proj"])
def test_replicated_mla_projection_does_not_gather_duplicate_tp_rows(
    converter, bridge_factory, monkeypatch, projection
):
    name = f"module.module.decoder.layers.3.self_attention.{projection}.weight"
    tensor = torch.nn.Parameter(torch.arange(64, dtype=torch.float32).reshape(4, 16))
    tensor.tensor_model_parallel = True
    monkeypatch.setattr(
        converter.dist,
        "all_gather",
        lambda *_args, **_kwargs: pytest.fail("Replicated MLA weight was gathered"),
    )
    gathered = converter.all_gather_param(name, tensor, duplicated_param_names={name})
    converted = _convert(converter, bridge_factory(_config()), name, gathered)
    torch.testing.assert_close(converted[0][1], tensor, rtol=0, atol=0)


def test_engine_expert_bucket_uses_v3_bridge_after_ep_gather(
    converter, bridge_factory, monkeypatch
):
    sender = _sender_methods(converter)()
    sender.bridge = bridge_factory(_config())
    sender.hf_config = sender.bridge.hf_config
    sender.tf_config = sender.bridge.config
    sender.quantization_config = None
    sender.fp8_direct_convert = False
    sender.is_pipeline_parallel_head = lambda: True
    monkeypatch.setattr(converter.mpu, "get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(
        converter.mpu, "get_expert_model_parallel_world_size", lambda: 2
    )
    names = [
        f"module.module.decoder.layers.4.mlp.experts.linear_fc1.weight{index}"
        for index in (0, 4)
    ]
    tensors = [
        torch.arange(128, dtype=torch.float32).reshape(8, 16) + index * 1000
        for index in range(2)
    ]
    waited = []

    def gather_names(outputs, _names, *, group):
        assert group == "ep"
        outputs[:] = [[name] for name in names]

    def gather_tensors(outputs, _tensor, *, group, async_op):
        assert group == "ep" and async_op
        for output, expected in zip(outputs, tensors, strict=True):
            output.copy_(expected)
        return SimpleNamespace(wait=lambda: waited.append(True))

    monkeypatch.setattr(converter.dist, "all_gather_object", gather_names)
    monkeypatch.setattr(converter.dist, "all_gather", gather_tensors)
    received = {}
    sender._update_bucket_weights_from_distributed = (
        lambda _meta, weights: received.update(weights)
    )
    bucket = [(names[0], tensors[0])]
    sender._update_bucket_expert_weights_from_distributed(SimpleNamespace(), bucket)

    assert bucket == [] and waited == [True]
    assert len(received) == 4
    for expert_id, tensor in zip((0, 4), tensors, strict=True):
        for projection, expected in zip(("gate", "up"), tensor.chunk(2), strict=True):
            torch.testing.assert_close(
                received[
                    f"model.layers.4.mlp.experts.{expert_id}.{projection}_proj.weight"
                ],
                expected,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize(
    "use_lora,quantization", [(True, None), (False, {"quant_method": "fp8"})]
)
def test_engine_rejects_unsupported_v3_modes_before_creating_nccl_group(
    converter, use_lora, quantization
):
    sender = _sender_methods(converter)()
    sender.hf_config = _config()
    sender.config = SimpleNamespace(use_lora=use_lora)
    sender.quantization_config = quantization
    sender.fp8_direct_convert = False
    # Deliberately no rendezvous/config attributes: failure must precede their use.
    with pytest.raises(NotImplementedError, match="unquantized full-model"):
        sender._init_weight_update_from_distributed(SimpleNamespace(type="xccl"))
